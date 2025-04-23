from flask import Flask,request,jsonify,render_template
import requests
import pickle

#Loading the model
with open('ensemble_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

#initalizing  the flask
app = Flask(__name__, template_folder="templates")

#API keys
FACT_CHECK_API_KEY = "AIzaSyAQ_ML6c7ywdZFO1EiX0E4ZXwH-pCbaX3I"
CUSTOM_SEARCH_API_KEY = "AIzaSyCL2CfKR1s7AT3qwn_lIltu3a5qemfT9Mw"
CUSTOM_SEARCH_ENGINE_ID = "8552f48287d2d4c30"

#News Predicting function
def predict_news(news_text):
    pt=[news_text.lower()]
    vt=vectorizer.transform(pt)
    prediction = model.predict(vt)[0]
    return "Fake News" if prediction==0 else "Real News"

#Google custom  api Function
def google_custom_search(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={CUSTOM_SEARCH_API_KEY}&cx={CUSTOM_SEARCH_ENGINE_ID}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        search_results = []
        if 'items' in data:
            for item in data['items'][:2]: 
                search_results.append({
                    "title": item['title'],
                    "link": item['link'],
                    "snippet": item['snippet']
                })
        return search_results
    else:
        return []
    
#Fact Check api Function
def fact_check(news_text):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={news_text}&key={FACT_CHECK_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        fact_check_results = []
        if "claims" in data and data["claims"]:
            for claim in data["claims"][:2]:  # Limit to first 2 claims
                review = claim.get("claimReview", [{}])[0]  # Get first review if available
                fact_check_results.append({
                    "text": claim.get("text", "No text available"),
                    "review": review.get("textualRating", "No rating available"),
                    "url": review.get("url", "No source")
                })
        return fact_check_results
    else:
        return []
    
#Prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    news_text = data.get("news_text", "")
    
    if not news_text:
        return jsonify({"error": "No news text provided"}), 400
    
    # Model prediction
    prediction = predict_news(news_text)
    
    # sources from Google search API
    search_results = google_custom_search(news_text)
    
    # fact-checking api
    fact_check_results = fact_check(news_text)
    
    return jsonify({
        "prediction": prediction,
        "search_results": search_results,
        "fact_check_results": fact_check_results
    })

#Home Page
@app.route('/')
def home():
    return render_template('index.html')



if __name__=='__main__':
    app.run(debug=True)