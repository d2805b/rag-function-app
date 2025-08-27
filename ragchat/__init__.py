import azure.functions as func
import logging
import json
import os
import openai
import requests

# Azure OpenAI の設定
openai.api_type = "azure"
openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]
openai.api_version = "2024-02-15-preview"
deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT"]

# Azure Search の設定
search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
search_key = os.environ["AZURE_SEARCH_KEY"]
search_index = os.environ["AZURE_SEARCH_INDEX_NAME"]

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('RAGチャット関数が呼び出されました')

    try:
        # 入力の取り方を強化（JSON / クエリ両対応）
        try:
            req_body = req.get_json()
        except ValueError:
            req_body = None
        question = (req_body or {}).get("question") or req.params.get("question")

        if not question:
            return func.HttpResponse(
                json.dumps({"error": "BadRequest: 'question' が必要です。"}, ensure_ascii=False),
                mimetype="application/json",
                status_code=400
            )

        # ---- ここから検索 ----
        search_url = f"{search_endpoint}/indexes/{search_index}/docs/search?api-version=2023-07-01-preview"
        headers = {"Content-Type": "application/json", "api-key": search_key}
        payload = {"search": question, "top": 3}
        search_response = requests.post(search_url, headers=headers, json=payload)
        search_response.raise_for_status()
        results = search_response.json()

        context_list, source_list = [], []
        for doc in results.get("value", []):
            score = doc.get("@search.score", 0)
            file_name = doc.get("metadata_storage_name", "unknown")
            content = doc.get("content", "")
            if content:
                context_list.append(f"[{file_name}] {content}")
                source_list.append({"score": round(score, 2), "source": file_name})

        combined_context = "\n\n".join(context_list)

        # ---- OpenAI 呼び出し ----
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2024-02-15-preview",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )

        system_prompt = (
            "あなたは社内データに基づいて回答するAIアシスタントです。"
            "参照情報に根拠がない場合は『情報が見つかりませんでした』と答えてください。"
            "参照情報を使わなかった場合は引用を出力しない前提で応答します。"
        )

        user_content = f"質問: {question}"
        if combined_context:
            user_content += f"\n\n参照情報:\n{combined_context}"

        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        answer = response.choices[0].message.content

        # 教師データ未使用なら sources は空で返す（フロントは非表示になる）
        sources_output = source_list if source_list else []

        return func.HttpResponse(
            json.dumps({"answer": answer, "sources": sources_output}, ensure_ascii=False),
            mimetype="application/json",
            status_code=200,
        )

    except requests.HTTPError as http_err:
        logging.exception("Search API エラー")
        return func.HttpResponse(
            json.dumps({"error": f"SearchError: {http_err.response.status_code} {http_err.response.text}"}, ensure_ascii=False),
            mimetype="application/json",
            status_code=502,
        )
    except KeyError as ke:
        logging.exception("環境変数不足")
        return func.HttpResponse(
            json.dumps({"error": f"ConfigError: 環境変数が不足しています: {str(ke)}"}, ensure_ascii=False),
            mimetype="application/json",
            status_code=500,
        )
    except Exception as e:
        logging.exception("予期せぬエラー")
        return func.HttpResponse(
            json.dumps({"error": f"InternalError: {str(e)}"}, ensure_ascii=False),
            mimetype="application/json",
            status_code=500,
        )
