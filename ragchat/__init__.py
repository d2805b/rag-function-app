import azure.functions as func
import logging
import json
import os
import openai
import requests

# ===== Azure OpenAI の設定 =====
openai.api_type = "azure"
openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]
openai.api_version = "2024-02-15-preview"
deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT"]

# ===== Azure Search の設定 =====
search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
search_key = os.environ["AZURE_SEARCH_KEY"]
search_index = os.environ["AZURE_SEARCH_INDEX_NAME"]

# ===== ガード設定（環境変数でチューニング可能） =====
# 最小関連度スコア（@search.score の閾値）
MIN_SCORE = float(os.environ.get("MIN_RELEVANCE_SCORE", "1.0"))
# 任意：質問がこの語群を一つも含まなければ対象外にする（カンマ区切り）
ALLOWED_TOPICS = [x.strip() for x in os.environ.get("ALLOWED_TOPICS", "").split(",") if x.strip()]

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('RAGチャット関数が呼び出されました')

    try:
        # ---- 入力取得（JSON or クエリ） ----
        try:
            req_body = req.get_json()
        except ValueError:
            req_body = None
        question = (req_body or {}).get("question") or req.params.get("question")

        if not question:
            return _resp({"error": "BadRequest: 'question' が必要です。"}, 400)

        # ---- 任意：トピック・ゲート（完全に対象外は即終了） ----
        if ALLOWED_TOPICS:
            if not any(k in question for k in ALLOWED_TOPICS):
                return _resp({
                    "answer": "この質問は対象の社内データ領域外のため回答できません。",
                    "sources": []
                }, 200)

        # ---- Cognitive Search で検索 ----
        search_url = f"{search_endpoint}/indexes/{search_index}/docs/search?api-version=2023-07-01-preview"
        headers = {"Content-Type": "application/json", "api-key": search_key}
        payload = {
            "search": question,
            "top": 5  # 広めに取得して後段でフィルタ
            # 必要に応じて以下を有効化
            # "queryType": "semantic",
            # "semanticConfiguration": "default"
        }
        search_response = requests.post(search_url, headers=headers, json=payload, timeout=30)
        search_response.raise_for_status()
        results = search_response.json()

        # ---- 関連度ゲート：@search.score でフィルタ ----
        context_list = []
        source_list = []
        for doc in results.get("value", []):
            try:
                score = float(doc.get("@search.score", 0))
            except (TypeError, ValueError):
                score = 0.0
            if score < MIN_SCORE:
                continue  # 閾値未満は捨てる

            file_name = doc.get("metadata_storage_name", "unknown")
            content = doc.get("content", "")
            if not content:
                continue

            context_list.append(f"[{file_name}] {content}")
            source_list.append({
                "score": round(score, 2),
                "source": file_name
            })

        # ---- 根拠ゼロならモデルは呼ばない ----
        if not context_list:
            return _resp({
                "answer": "申し訳ありません。この質問は対象の社内データに根拠が見つからないため回答できません。",
                "sources": []
            }, 200)

        combined_context = "\n\n".join(context_list)

        # ---- OpenAI へ問い合わせ（強めの system プロンプト） ----
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2024-02-15-preview",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
        )

        system_prompt = (
            "あなたは社内ドキュメントに基づいてのみ回答するアシスタントです。"
            "提供された参照情報に根拠がない場合は回答してはいけません。"
            "その場合は『この質問は対象の社内データに根拠が見つからないため回答できません』とだけ答えてください。"
            "一般的な知識や推測で補完すること、参照情報に無い引用や外部情報を出すことは禁止です。"
        )

        user_content = f"質問: {question}\n\n参照情報:\n{combined_context}"

        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1,   # 逸脱抑制
            max_tokens=800
        )
        answer = response.choices[0].message.content

        # 教師データ未使用時は空配列（今回は根拠ありで来ている想定だが明示）
        sources_output = source_list if source_list else []

        return _resp({"answer": answer, "sources": sources_output}, 200)

    except requests.HTTPError as http_err:
        logging.exception("Search API エラー")
        status = http_err.response.status_code if http_err.response is not None else 502
        return _resp({"error": f"SearchError: {status}"}, 502)

    except KeyError as ke:
        logging.exception("環境変数不足")
        return _resp({"error": f"ConfigError: 環境変数が不足しています: {str(ke)}"}, 500)

    except Exception as e:
        logging.exception("予期せぬエラー")
        return _resp({"error": f"InternalError: {str(e)}"}, 500)


# 共通レスポンス
def _resp(obj: dict, status: int) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(obj, ensure_ascii=False),
        mimetype="application/json",
        status_code=status
    )
