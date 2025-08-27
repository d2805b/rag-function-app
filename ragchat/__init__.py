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

        # Cognitive Search で検索
        search_url = f"{search_endpoint}/indexes/{search_index}/docs/search?api-version=2023-07-01-preview"
        headers = {
            "Content-Type": "application/json",
            "api-key": search_key
        }
        payload = {
            "search": question,
            "top": 3
        }
        search_response = requests.post(search_url, headers=headers, json=payload)
        results = search_response.json()

        context_list = []
        source_list = []

        for doc in results.get("value", []):
            score = doc.get("@search.score", 0)
            file_name = doc.get("metadata_storage_name", "unknown")
            content = doc.get("content", "")
            context_list.append(f"[{file_name}] {content}")
            source_list.append({
                "score": round(score, 2),
                "source": file_name
            })

        combined_context = "\n\n".join(context_list)

        # OpenAI に問い合わせ
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
                        "あなたは社内データに基づいて回答するAIアシスタントです。"
                        "教師データ（以下の参照情報）に根拠がない場合は回答してはいけません。"
                        "教師データに基づかない一般的な知識や推測では回答せず、"
                        "必ず「情報が見つかりませんでした」と答えてください。"
                        "また、参照情報を使用しなかった場合は引用情報を出力しないでください。"
                    )
                },
    {
        "role": "user",
        "content": f"質問: {question}\n\n参照情報:\n{combined_context}"
    }
]

            temperature=0.3,
            max_tokens=800
        )
        answer = response.choices[0].message.content

        sources_output = source_list if source_list else []

        return func.HttpResponse(
            json.dumps({
                "answer": answer,
                "sources": sources_output
            }, ensure_ascii=False),
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
