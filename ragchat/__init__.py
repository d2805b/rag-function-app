
import azure.functions as func
import logging
import json
import os
import openai
import requests

app = func.FunctionApp()

# Azure OpenAI の設定
openai.api_type = "azure"
openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]
openai.api_version = "2024-02-15-preview"  # 必要に応じて調整
deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT"]

# Azure Search の設定
search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
search_key = os.environ["AZURE_SEARCH_KEY"]
search_index = os.environ["AZURE_SEARCH_INDEX_NAME"]

@app.route(route="main", auth_level=func.AuthLevel.ANONYMOUS)
def ragchat(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('RAGチャット関数が呼び出されました')

    try:
        req_body = req.get_json()
        question = req_body.get("question")

        if not question:
            raise ValueError("質問（question）がリクエストに含まれていません")

        # Azure Cognitive Search で検索
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

        # 検索結果を整形
        context_texts = [
            (doc["@search.score"], doc["content"] if "content" in doc else str(doc))
            for doc in results.get("value", [])
        ]
        context = "\n".join([f"[スコア: {score:.2f}] {text}" for score, text in context_texts])

        # OpenAI に質問
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2024-02-15-preview",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
        )

        response = client.chat.completions.create(
            model=deployment_name,  # デプロイ名を指定
            messages=[
                {"role": "system", "content": "以下の情報に基づいて..."},
                {"role": "user", "content": f"質問: {question}\n\n参照情報:\n{context}"}
            ],
            temperature=0.3,
            max_tokens=800
        )

        answer = response.choices[0].message.content

        return func.HttpResponse(
            json.dumps({"answer": answer}),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error("エラーが発生しました", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
