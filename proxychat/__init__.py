import azure.functions as func
import os, json, requests

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
        question = body.get("question")
        if not question:
            return func.HttpResponse(json.dumps({"error":"質問が空です"}), status_code=400, mimetype="application/json")

        rag_url = os.environ["RAGCHAT_URL"]         
        rag_code = os.environ.get("RAGCHAT_CODE","")
        url = f"{rag_url}?code={rag_code}" if rag_code else rag_url

        resp = requests.post(url, json={"question": question}, timeout=30)
        return func.HttpResponse(resp.text, status_code=resp.status_code, mimetype="application/json")

    except Exception as e:
        return func.HttpResponse(json.dumps({"error": str(e)}), status_code=500, mimetype="application/json")
