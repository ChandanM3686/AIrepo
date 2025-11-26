# s.py - Streamlit version of app.py (updated to prefer 4 recommendations & progressive relaxation)
import streamlit as st
import requests
import os
import time
import re
import colorsys
from io import BytesIO
from urllib.parse import quote

try:
    from PIL import Image
except ImportError:
    Image = None

# === Put your SerpAPI key here or set as environment variable SERPAPI_KEY ===
SERPAPI_KEY = os.getenv("SERPAPI_KEY") or "a3c301766fd3f33d0f8b2c76c10d2beccd3ccdfbad2b31f5b80d0a50f3f48b7c"
SERPAPI_URL = "https://serpapi.com/search.json"

# Tunables
IMMERSIVE_CACHE_TTL = 300  # seconds
if 'immersive_cache' not in st.session_state:
    st.session_state.immersive_cache = {}
RESULTS_TO_DISPLAY = 4
RESULTS_MIN_DISPLAY = 4
# increased multiplier to fetch more candidates for relaxation steps
RESULTS_FETCH_MULTIPLIER = 10
IMAGE_COLOR_CACHE_TTL = 900
if 'image_color_cache' not in st.session_state:
    st.session_state.image_color_cache = {}
MAX_IMAGE_BYTES = 600 * 1024

CATEGORY_MAP = {
    "formal": "formal wear office",
    "ethnic": "ethnic traditional",
    "party": "party wear",
    "sport": "sports active",
    "casual": "casual everyday",
}

def build_query(payload):
    parts = []
    key = (payload.get("category_key") or payload.get("category") or "").strip().lower()
    mapped = CATEGORY_MAP.get(key)
    if mapped:
        parts.append(mapped)
    else:
        if payload.get("category") and payload.get("category").strip():
            parts.append(payload.get("category").strip())
    for k in ("items", "budget", "gender", "size"):
        v = payload.get(k)
        if v and str(v).strip():
            parts.append(str(v).strip())
    query = " ".join(parts)
    query = " ".join(query.split())
    if not query:
        query = (payload.get("items") or payload.get("category") or "").strip()
    return query

def normalize_tokens(text):
    if not text:
        return []
    return [
        tok.lower()
        for tok in str(text).replace("/", " ").replace("-", " ").split()
        if len(tok) > 2
    ]

def keyword_constraints(payload):
    must = []
    prefer = []
    category_tokens = normalize_tokens(payload.get("category"))
    if category_tokens:
        # prioritize last token (often product type e.g. "shirt")
        must.append(category_tokens[-1])
        prefer.extend(tok for tok in category_tokens if tok not in must)
    item_tokens = normalize_tokens(payload.get("items"))
    for tok in item_tokens:
        if tok not in must:
            prefer.append(tok)
    # remove duplicates while preserving order
    seen = set()
    def dedupe(seq):
        for token in seq:
            if token and token not in seen:
                seen.add(token)
                yield token
    must = list(dedupe(must))
    prefer = [tok for tok in dedupe(prefer) if tok not in must]
    return must, prefer

def prioritize_results(results, must_words, prefer_words):
    if not results or (not must_words and not prefer_words):
        return results
    scored = []
    fallback = []
    for item in results:
        title = (item.get("title") or "").lower()
        if not title:
            fallback.append(item)
            continue
        if must_words and not all(word in title for word in must_words):
            fallback.append(item)
            continue
        score = sum(1 for word in prefer_words if word in title)
        scored.append((score, item))
    if not scored:
        # If no exact must-word matches, fall back to prefer-word sorting across all results
        fallback_scored = []
        for item in results:
            title = (item.get("title") or "").lower()
            score = sum(1 for word in prefer_words if word in title)
            fallback_scored.append((score, item))
        fallback_scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in fallback_scored]
    scored.sort(key=lambda x: x[0], reverse=True)
    prioritized = [item for _, item in scored]
    if must_words:
        return prioritized
    return prioritized + fallback

COLOR_WORDS = {
    "white", "black", "blue", "navy", "red", "pink", "green", "olive", "yellow",
    "gold", "silver", "gray", "grey", "brown", "maroon", "purple", "violet",
    "beige", "cream", "tan", "orange", "teal", "turquoise", "burgundy"
}
COLOR_ALIASES = {
    "navy": "blue",
    "teal": "blue",
    "turquoise": "blue",
    "aqua": "blue",
    "sky": "blue",
    "burgundy": "maroon",
    "violet": "purple",
    "cream": "beige",
    "tan": "beige",
    "grey": "gray",
}
COLOR_TBS_MAP = {
    "black": "black",
    "white": "white",
    "blue": "blue",
    "red": "red",
    "pink": "pink",
    "green": "green",
    "yellow": "yellow",
    "orange": "orange",
    "brown": "brown",
    "purple": "purple",
    "gray": "gray",
    "beige": "beige",
}

def extract_target_colors(payload):
    tokens = []
    # Check dedicated color field first
    color_field = payload.get("color", "").strip()
    if color_field:
        tokens.append(color_field)
    # Also extract from items and category
    for key in ("items", "category"):
        tokens.extend(normalize_tokens(payload.get(key)))
    colors = []
    for tok in tokens:
        if tok in COLOR_WORDS:
            colors.append(tok)
    seen = set()
    result = []
    for color in colors:
        normalized = normalize_color_token(color)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result

def normalize_color_token(token):
    if not token:
        return ""
    token = token.lower()
    token = COLOR_ALIASES.get(token, token)
    return token

def parse_budget_value(text):
    if not text:
        return None
    nums = re.findall(r"\d+", str(text))
    if not nums:
        return None
    try:
        values = [int(n) for n in nums]
        return max(values)
    except ValueError:
        return None

def parse_price_text(price_text):
    if not price_text:
        return None
    nums = re.findall(r"\d+", price_text.replace(",", ""))
    if not nums:
        return None
    try:
        return int(nums[0])
    except ValueError:
        return None

def filter_by_color_and_price(results, payload, preset_colors=None, strict_color=True, strict_budget=True):
    """
    strict_color=True: require color match (title or image)
    strict_budget=True: require price <= budget if budget specified
    If strict_color is False, color is used only to push items up in ranking (not strictly required)
    """
    target_colors = preset_colors if preset_colors is not None else extract_target_colors(payload)
    normalized_colors = [normalize_color_token(c) for c in target_colors]
    normalized_colors = [c for c in normalized_colors if c]
    budget_limit = parse_budget_value(payload.get("budget"))
    if not normalized_colors and budget_limit is None:
        return results

    filtered = []
    budget_only = []  # items within budget but no color match

    for item in results:
        title = (item.get("title") or "").lower()
        raw = item.get("raw") or {}
        price_val = raw.get("extracted_total") or raw.get("extracted_price")
        if price_val is None:
            price_val = parse_price_text(item.get("price_text"))

        # budget check
        if budget_limit is not None and strict_budget:
            if price_val is None:
                # skip if strict budget required and price unknown
                continue
            if price_val > budget_limit:
                continue

        # color matching
        color_matches = False
        if normalized_colors:
            # Check title first
            for color in normalized_colors:
                if color in title:
                    color_matches = True
                    break
                # common variants
                if color == "blue":
                    variants = ["blue", "navy", "powder blue", "royal blue", "sky blue", "teal", "turquoise", "azure"]
                    if any(v in title for v in variants):
                        color_matches = True
                        break
                elif color == "black":
                    variants = ["black", "ebony", "charcoal", "onyx", "jet black"]
                    if any(v in title for v in variants):
                        color_matches = True
                        break
                elif color == "white":
                    variants = ["white", "ivory", "cream", "pearl", "snow"]
                    if any(v in title for v in variants):
                        color_matches = True
                        break
                elif color == "red":
                    variants = ["red", "crimson", "scarlet", "burgundy", "maroon", "cherry"]
                    if any(v in title for v in variants):
                        color_matches = True
                        break
                elif color == "green":
                    variants = ["green", "emerald", "olive", "mint", "forest"]
                    if any(v in title for v in variants):
                        color_matches = True
                        break
            # If not matched in title, try image color
            if not color_matches:
                color_matches = image_matches_color(item, normalized_colors)

            if color_matches:
                filtered.append(item)
            else:
                # keep for possible relaxed return (if strict_color False we'll include later)
                budget_only.append(item)
        else:
            # No color filter specified
            if budget_limit is not None and strict_budget:
                # include if within budget
                if price_val is None:
                    continue
                if price_val <= budget_limit:
                    filtered.append(item)
            else:
                filtered.append(item)

    if normalized_colors:
        if strict_color:
            return filtered
        else:
            # relaxed: return color-matches first, then budget-only, then the rest
            # dedupe while preserving order
            merged = []
            seen_links = set()
            def add_list(lst):
                for it in lst:
                    key = it.get("link") or it.get("title") or str(id(it))
                    if key not in seen_links:
                        seen_links.add(key)
                        merged.append(it)
            add_list(filtered)
            add_list(budget_only)
            # finally add all original results as last resort
            add_list(results)
            return merged
    else:
        if budget_limit is not None:
            return filtered
        return results

def image_matches_color(item, colors):
    if not colors:
        return True
    if Image is None:
        return False
    url = item.get("thumbnail") or item.get("raw", {}).get("thumbnail")
    approx = approximate_image_color(url)
    if not approx:
        return False
    return approx in set(colors)

def approximate_image_color(url):
    if not url or Image is None:
        return None
    now = time.time()
    cached = st.session_state.image_color_cache.get(url)
    if cached and now - cached[1] < IMAGE_COLOR_CACHE_TTL:
        return cached[0]
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        content = resp.content
        if len(content) > MAX_IMAGE_BYTES:
            st.session_state.image_color_cache[url] = (None, now)
            return None
        img = Image.open(BytesIO(content)).convert("RGB")
        img = img.resize((40, 40))
        pixels = list(img.getdata())
        r = sum(p[0] for p in pixels) / len(pixels)
        g = sum(p[1] for p in pixels) / len(pixels)
        b = sum(p[2] for p in pixels) / len(pixels)
        raw_color = map_rgb_to_color((r, g, b))
        color_name = normalize_color_token(raw_color)
        st.session_state.image_color_cache[url] = (color_name, now)
        return color_name
    except Exception as exc:
        print(f"[color] failed to analyze image {url}: {exc}")
        st.session_state.image_color_cache[url] = (None, now)
        return None

def map_rgb_to_color(rgb):
    r, g, b = [v / 255.0 for v in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h_deg = h * 360
    if v < 0.2:
        return "black"
    if v > 0.92 and s < 0.15:
        return "white"
    if v > 0.8 and s < 0.25:
        return "gray"
    if s < 0.2:
        return "beige"
    if 0 <= h_deg < 20 or h_deg >= 340:
        return "red"
    if 20 <= h_deg < 45:
        return "orange"
    if 45 <= h_deg < 70:
        return "yellow"
    if 70 <= h_deg < 165:
        return "green"
    if 165 <= h_deg < 210:
        return "teal"
    if 210 <= h_deg < 255:
        return "blue"
    if 255 <= h_deg < 300:
        return "purple"
    if 300 <= h_deg < 340:
        return "pink"
    return "gray"

def count_color_matches(results, colors):
    if not colors:
        return 0
    normalized = [normalize_color_token(c) for c in colors if c]
    total = 0
    for item in results:
        title = (item.get("title") or "").lower()
        if any(color in title for color in normalized):
            total += 1
            continue
        if image_matches_color(item, normalized):
            total += 1
    return total

def merge_unique_results(*result_lists):
    seen = set()
    merged = []
    for lst in result_lists:
        for item in lst or []:
            key = item.get("link") or item.get("title") or str(id(item))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged

def build_color_filter_param(colors):
    if not colors:
        return None
    for color in colors:
        normalized = normalize_color_token(color)
        if normalized and normalized in COLOR_TBS_MAP:
            return f"mr:1,col:{COLOR_TBS_MAP[normalized]}"
    return None

def normalize_price(price_field):
    if price_field is None:
        return ""
    if isinstance(price_field, dict):
        return price_field.get("raw") or str(price_field.get("value") or "")
    return str(price_field)

def extract_shopping_results(data, num_results=6):
    sr = data.get("shopping_results") or data.get("product_results") or data.get("inline_products") or []
    results = []
    for item in sr:
        title = item.get("title") or item.get("product_title") or item.get("name") or ""
        price = normalize_price(item.get("price") or item.get("extracted_price") or item.get("product_price"))
        thumbnail = item.get("thumbnail") or item.get("image") or item.get("product_image") or ""
        link = item.get("link") or item.get("product_link") or item.get("serpapi_product_url") or ""
        merchant_link = item.get("product_link") or item.get("source_link") or ""
        immersive_token = item.get("immersive_product_page_token") or ""
        source = ""
        if isinstance(item.get("source"), dict):
            source = item.get("source").get("name") or ""
        else:
            source = item.get("merchant") or item.get("store") or item.get("source") or ""
        if thumbnail and thumbnail.startswith("//"):
            thumbnail = "https:" + thumbnail
        if link and link.startswith("/"):
            link = "https://www.google.com" + link
        results.append({
            "title": title,
            "price_text": price,
            "thumbnail": thumbnail,
            "link": link,
            "merchant_link": merchant_link,
            "immersive_token": immersive_token,
            "source": source,
            "raw": item
        })
        if len(results) >= num_results:
            break
    return results

def serpapi_search_shopping(query, num_results=6, country="in", language="en", max_retries=3, extra_params=None):
    if not query or not str(query).strip():
        raise ValueError("Empty query passed to serpapi_search_shopping()")
    params = {
        "engine": "google",
        "q": query,
        "tbm": "shop",
        "hl": language,
        "gl": country,
        "api_key": SERPAPI_KEY,
    }
    if extra_params:
        params.update(extra_params)
    backoff = 1.0
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[serpapi] attempt {attempt} -> query: {query!r} params={extra_params}")
            resp = requests.get(SERPAPI_URL, params=params, timeout=12)
            status = resp.status_code
            content_type = resp.headers.get("content-type", "")
            print(f"[serpapi] status={status} content-type={content_type}")
            if status >= 400:
                body = None
                try:
                    body = resp.json()
                except Exception:
                    body = resp.text
                if status == 429 or (500 <= status < 600):
                    print(f"[serpapi] retryable error ({status}), body={body}")
                    time.sleep(backoff)
                    backoff *= 2
                    last_exc = requests.HTTPError(f"SerpAPI status {status}: {body}")
                    continue
                raise requests.HTTPError(f"SerpAPI returned status {status}: {body}")
            data = resp.json()
            results = extract_shopping_results(data, num_results=num_results)
            return results, data
        except requests.Timeout as te:
            print(f"[serpapi] timeout on attempt {attempt}: {te}")
            last_exc = te
            time.sleep(backoff)
            backoff *= 2
            continue
        except requests.HTTPError as he:
            print(f"[serpapi] HTTP error: {he}")
            raise
        except Exception as e:
            print(f"[serpapi] unexpected error on attempt {attempt}: {e}")
            last_exc = e
            time.sleep(backoff)
            backoff *= 2
            continue
    raise RuntimeError(f"SerpAPI search failed after {max_retries} attempts. Last error: {last_exc!r}")

def cached_direct_store_url(token, language="en", country="in"):
    if not token:
        return None
    now = time.time()
    cached = st.session_state.immersive_cache.get(token)
    if cached:
        link, ts = cached
        if now - ts < IMMERSIVE_CACHE_TTL:
            return link
    params = {
        "engine": "google_immersive_product",
        "page_token": token,
        "hl": language,
        "gl": country,
        "api_key": SERPAPI_KEY,
    }
    try:
        resp = requests.get(SERPAPI_URL, params=params, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        stores = (data.get("product_results") or {}).get("stores") or []
        for store in stores:
            link = store.get("link")
            if link:
                st.session_state.immersive_cache[token] = (link, now)
                return link
    except Exception as exc:
        print(f"[immersive] Failed to load stores for token={token[:6]}...: {exc}")
    st.session_state.immersive_cache[token] = (None, now)
    return None

def get_recommendations(data):
    """
    Progressive search logic:
    1) Build query and try a strict search (color tbs, strict filtering)
    2) If results < min, do a broader search without tbs (relaxed color)
    3) If still < min, try color-augmented queries and prefer-word based fetch
    4) Final: relaxed filtering (color optional) and return top RESULTS_TO_DISPLAY
    """
    query = build_query(data)
    print("DEBUG QUERY ===>", query)
    if not query or not str(query).strip():
        return {"error": "Empty search query built from input"}, None

    target_colors = extract_target_colors(data)
    must_words, prefer_words = keyword_constraints(data)

    # Try param with color tbs (if available)
    color_filter_param = build_color_filter_param(target_colors)
    extra_params = {}
    if color_filter_param:
        extra_params["tbs"] = color_filter_param
    if not extra_params:
        extra_params = None

    # fetch a lot of candidates for ranking & relaxation
    fetch_count = RESULTS_TO_DISPLAY * RESULTS_FETCH_MULTIPLIER

    # 1) Primary: strict search (if color/budget specified)
    try:
        results_primary, raw_primary = serpapi_search_shopping(query, num_results=fetch_count, extra_params=extra_params)
    except requests.HTTPError as e:
        return {"error": "SerpAPI HTTP error", "details": str(e), "query": query}, None
    except Exception as e:
        print("FULL ERROR:", repr(e))
        return {"error": "SerpAPI request failed", "details": repr(e), "query": query}, None

    all_results = list(results_primary)
    # check color hits - if too few, we'll broaden in steps
    color_hits = count_color_matches(all_results, target_colors)

    # If color requested but few color hits, do an augmented fetch (color as prefix to query)
    if target_colors and color_hits < RESULTS_MIN_DISPLAY:
        augmented_query = f"{' '.join(target_colors)} {query}".strip()
        try:
            extra_results, _ = serpapi_search_shopping(augmented_query, num_results=fetch_count, extra_params=extra_params)
            all_results = merge_unique_results(all_results, extra_results)
        except Exception as exc:
            print(f"[recommend] color-augmented fetch failed: {exc}")

    # If still not enough, try without tbs (remove color param) to widen
    if target_colors and color_hits < RESULTS_MIN_DISPLAY:
        try:
            extra_results_no_tbs, _ = serpapi_search_shopping(query, num_results=fetch_count, extra_params=None)
            all_results = merge_unique_results(all_results, extra_results_no_tbs)
        except Exception as exc:
            print(f"[recommend] fetch without tbs failed: {exc}")

    # Also try queries built from must/prefer words (two variants)
    if must_words or prefer_words:
        q1 = " ".join(must_words + prefer_words + [query])
        try:
            qr1_results, _ = serpapi_search_shopping(q1, num_results=fetch_count // 2, extra_params=None)
            all_results = merge_unique_results(all_results, qr1_results)
        except Exception as exc:
            print(f"[recommend] must/prefer augmented fetch failed: {exc}")

    # priorize using keyword constraints
    keyword_results = prioritize_results(all_results, must_words, prefer_words)

    # 2) Strict filter first (color + budget enforced)
    filtered_strict = filter_by_color_and_price(keyword_results, data, preset_colors=target_colors, strict_color=True, strict_budget=True)

    # If strict gives enough:
    if len(filtered_strict) >= RESULTS_MIN_DISPLAY:
        final = filtered_strict[:RESULTS_TO_DISPLAY]
        return {"query": query, "count": len(final), "results": final}, raw_primary

    # 3) Relax budget strictness if budget is provided (allow unknown-priced items)
    filtered_budget_relaxed = filter_by_color_and_price(keyword_results, data, preset_colors=target_colors, strict_color=True, strict_budget=False)
    if len(filtered_budget_relaxed) >= RESULTS_MIN_DISPLAY:
        final = filtered_budget_relaxed[:RESULTS_TO_DISPLAY]
        return {"query": query, "count": len(final), "results": final}, raw_primary

    # 4) Relax color requirement (color becomes preference not requirement)
    filtered_color_relaxed = filter_by_color_and_price(keyword_results, data, preset_colors=target_colors, strict_color=False, strict_budget=False)
    if len(filtered_color_relaxed) >= RESULTS_MIN_DISPLAY:
        final = filtered_color_relaxed[:RESULTS_TO_DISPLAY]
        return {"query": query, "count": len(final), "results": final}, raw_primary

    # 5) As a last resort, return top RESULTS_TO_DISPLAY from prioritized results (may not match color/budget)
    final = keyword_results[:RESULTS_TO_DISPLAY]
    return {"query": query, "count": len(final), "results": final}, raw_primary

# Streamlit UI
st.set_page_config(page_title="Fashion Recommender", page_icon="👕", layout="wide")

st.title("👕 Fashion Recommender (demo)")
st.markdown("Fill fields and click **Get recommendations**")

# Check if viewing a product detail
if 'view_product' in st.query_params and st.query_params['view_product'] == 'true':
    product_data = st.session_state.get('current_product')
    if product_data:
        st.button("← Back to results", on_click=lambda: st.query_params.clear())
        col1, col2 = st.columns([1, 1])
        with col1:
            if product_data.get('thumbnail'):
                st.image(product_data['thumbnail'], use_container_width=True)
        with col2:
            st.header(product_data.get('title', 'Product'))
            st.subheader(product_data.get('price_text', ''))
            st.write(f"**Sold on:** {product_data.get('source', 'N/A')}")
            token = product_data.get('immersive_token', '')
            merchant_link = product_data.get('merchant_link', '')
            link = product_data.get('link', '')
            direct_link = cached_direct_store_url(token) if token else None
            buy_url = direct_link or merchant_link or link
            if buy_url:
                st.markdown(f'<a href="{buy_url}" target="_blank" style="display:inline-block;margin-top:18px;padding:12px 16px;border-radius:8px;background:#10b981;color:white;text-decoration:none;font-weight:700;">Buy from {product_data.get("source", "merchant")}</a>', unsafe_allow_html=True)
            st.info("Selected product landing page. Only this product is shown here — no additional recommendations are displayed below.")
    else:
        st.error("Product data not found.")
        st.button("← Back to results", on_click=lambda: st.query_params.clear())
else:
    # Main form
    with st.form("recommendation_form"):
        category_key = st.selectbox(
            "Category key (or type a category)",
            options=["formal", "ethnic", "party", "sport", "casual"],
            index=0
        )
        category = st.text_input(
            "Category (eg. saree, shirt, suit) — also acts as fallback if not using category_key",
            placeholder="suit"
        )
        color = st.selectbox(
            "Color (optional - filters results by color)",
            options=["", "black", "white", "blue", "navy", "red", "pink", "green", "yellow",
                    "orange", "brown", "purple", "gray", "grey", "beige", "maroon", "teal"],
            index=0
        )
        items = st.text_input(
            "Items (specific item text)",
            placeholder="navy two piece suit"
        )
        budget = st.text_input(
            "Budget",
            placeholder="5000-10000"
        )
        gender = st.text_input(
            "Gender",
            placeholder="male / female"
        )
        size = st.text_input(
            "Size",
            placeholder="M"
        )
        submitted = st.form_submit_button("Get recommendations", type="primary")

    if submitted:
        # Combine color with items if color is selected
        items_with_color = items
        if color:
            items_with_color = f"{color} {items}".strip() if items else color

        payload = {
            "category_key": category_key,
            "category": category,
            "items": items_with_color,
            "color": color,  # Add dedicated color field
            "budget": budget,
            "gender": gender,
            "size": size,
        }
        with st.spinner("Searching..."):
            result, raw = get_recommendations(payload)

        if "error" in result:
            st.error(f"Error: {result.get('error')} - {result.get('details', '')}")
        else:
            # Show filter status if active
            target_colors = extract_target_colors(payload)
            budget_limit = parse_budget_value(payload.get("budget"))
            filter_info = []
            if target_colors:
                filter_info.append(f"Color: {', '.join(target_colors)}")
            if budget_limit:
                filter_info.append(f"Budget: ₹{budget_limit}")

            filter_text = f" | Filters: {', '.join(filter_info)}" if filter_info else ""
            if result['count'] == RESULTS_TO_DISPLAY:
                st.success(f"Found {result['count']} perfect recommendations for: \"{result['query']}\"{filter_text}")
            else:
                st.success(f"Found {result['count']} recommendations for: \"{result['query']}\"{filter_text}")

            if result['count'] == 0:
                warning_msg = "No recommendations found"
                if target_colors and budget_limit:
                    warning_msg = f"No recommendations found matching color: {', '.join(target_colors)} and budget: ₹{budget_limit}. Try adjusting your search criteria."
                elif target_colors:
                    warning_msg = f"No recommendations found matching color: {', '.join(target_colors)}. Try adjusting your search criteria."
                elif budget_limit:
                    warning_msg = f"No recommendations found within budget: ₹{budget_limit}. Try increasing your budget or adjusting your search criteria."
                st.warning(warning_msg)
            else:
                # Display results in a 2x2 grid (2 columns for 4 items)
                cols = st.columns(2)
                for idx, item in enumerate(result['results']):
                    col_idx = idx % 2
                    with cols[col_idx]:
                        with st.container():
                            if item.get('thumbnail'):
                                st.image(item['thumbnail'], use_container_width=True)
                            st.write(f"**{item.get('title', 'N/A')}**")
                            st.write(f"**Price:** {item.get('price_text', 'N/A')}")
                            st.write(f"**Source:** {item.get('source', 'N/A')}")

                            # Store product data for detail view
                            product_key = f"product_{idx}"
                            st.session_state[product_key] = item
                            st.session_state['current_product'] = item

                            # Create button to view product
                            token = item.get('immersive_token', '')
                            merchant_link = item.get('merchant_link', '')
                            link = item.get('link', '')
                            direct_link = cached_direct_store_url(token) if token else None
                            buy_url = direct_link or merchant_link or link

                            if buy_url:
                                st.markdown(
                                    f'<a href="{buy_url}" target="_blank" style="display:inline-block;margin-top:8px;padding:8px 10px;border-radius:6px;background:#0ea5e9;color:white;text-decoration:none;cursor:pointer;">View / Buy (open product page)</a>',
                                    unsafe_allow_html=True
                                )
                            st.divider()
