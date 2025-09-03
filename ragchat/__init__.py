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
        req_body = req.get_json()
        question = req_body.get("question")
        if not question:
            raise ValueError("質問（question）がリクエストに含まれていません")

        # 任意: 低スコア除外の閾値（未設定なら0）
        min_score = float(os.environ.get("SEARCH_MIN_SCORE", "0"))

        # Cognitive Search で検索
        search_url = f"{search_endpoint}/indexes/{search_index}/docs/search?api-version=2023-07-01-preview"
        headers = {"Content-Type": "application/json", "api-key": search_key}
        payload = {"search": question, "top": 3}

        search_response = requests.post(search_url, headers=headers, json=payload)
        if not search_response.ok:
            raise RuntimeError(f"Search API error: {search_response.status_code} {search_response.text}")

        results = search_response.json()

        context_list = []
        source_list = []

        for doc in results.get("value", []):
            score = float(doc.get("@search.score", 0))
            file_name = doc.get("metadata_storage_name", "unknown")
            content = (doc.get("content") or "").strip()
            # スコア閾値と空コンテンツ除外
            if score >= min_score and content:
                context_list.append(f"[{file_name}] {content}")
                source_list.append({"score": round(score, 2), "source": file_name})

        # 参照ゼロ → OpenAI を呼ばずに answer のみ返す
        if not context_list:
            return func.HttpResponse(
                json.dumps({"answer": "情報が見つかりませんでした。"}, ensure_ascii=False),
                mimetype="application/json",
                status_code=200
            )

        combined_context = "\n\n".join(context_list)

        # OpenAI に問い合わせ（参照情報に基づく回答のみ許可）
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2024-02-15-preview",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
        )
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "あなたは社内データ（参照情報）に基づいてのみ回答するアシスタントです。"
                        "参照情報に根拠がない場合は「情報が見つかりませんでした。」とだけ返してください。"
                        "一般知識や推測で補完しないでください。"
                    )
                },
                {"role": "user", "content": f"質問: {question}\n\n参照情報:\n{combined_context}"}
            ],
            temperature=1.2,
            max_tokens=800
        )
        answer = (response.choices[0].message.content or "").strip()

        # 情報なし回答なら sources を返さない
        NO_INFO_TEXT = os.environ.get("NO_INFO_TEXT", "情報が見つかりませんでした。")
        if answer == NO_INFO_TEXT:
            return func.HttpResponse(
                json.dumps({"answer": answer}, ensure_ascii=False),
                mimetype="application/json",
                status_code=200
            )

        # 参照ありのときだけ sources を同梱
        return func.HttpResponse(
            json.dumps({"answer": answer, "sources": source_list}, ensure_ascii=False),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error("エラーが発生しました", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": str(e)}, ensure_ascii=False),
            mimetype="application/json",
            status_code=500
        )
