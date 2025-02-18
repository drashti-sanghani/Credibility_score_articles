import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import pandas as pd
import os

class PageCredibilityEvaluator:
    """
    A robust URL evaluation class designed to assess the credibility of a webpage.
    The evaluation considers domain trust, content relevance, fact verification,
    potential bias, and citation analysis.
    """

    def __init__(self):
        # API Key for SerpAPI (Get this from the SerpAPI website)
        self.serp_api_key = "9167d02c0378e66cbfbc67823db1f976c35694211f9d93bb21b6bf8341878015"

        # Load necessary models for credibility analysis
        self.model_similarity = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.model_fake_news = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
        self.model_sentiment = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

    def extract_content(self, url: str) -> str:
        """ Extracts the content of a webpage by parsing its HTML. """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            return " ".join([p.text for p in soup.find_all("p")])  # Gather text from paragraph tags
        except requests.RequestException:
            return ""  # If there's an error, return an empty string

    def evaluate_domain_authority(self, url: str, content: str) -> int:
        """ Evaluates the domain authority score based on various criteria. """
        trust_scores = []

        # Check domain trust with Hugging Face's fake news model
        if content:
            try:
                trust_scores.append(self.assess_fake_news(content))
            except:
                pass

        # Return average score based on available sources or a default score
        return int(sum(trust_scores) / len(trust_scores)) if trust_scores else 50

    def assess_fake_news(self, content: str) -> int:
        """ Uses Hugging Face's fake news detection model to evaluate credibility. """
        if not content:
            return 50  # Return a neutral score if content is missing
        result = self.model_fake_news(content[:512])[0]  # Use first 512 characters for analysis
        return 100 if result["label"] == "REAL" else 30 if result["label"] == "FAKE" else 50

    def calculate_semantic_similarity(self, user_query: str, content: str) -> int:
        """ Measures how relevant the webpage content is to the user's query using semantic similarity. """
        if not content:
            return 0
        return int(util.pytorch_cos_sim(self.model_similarity.encode(user_query), self.model_similarity.encode(content)).item() * 100)

    def perform_fact_check(self, content: str) -> int:
        """ Verifies the content using Google's Fact Check API. """
        if not content:
            return 50
        api_url = f"https://toolbox.google.com/factcheck/api/v1/claimsearch?query={content[:200]}"
        try:
            response = requests.get(api_url)
            data = response.json()
            return 80 if "claims" in data and data["claims"] else 40
        except:
            return 50  # If fact-checking fails, return a neutral score

    def analyze_citations(self, url: str) -> int:
        """ Checks the number of Google Scholar citations for a given URL using SerpAPI. """
        params = {"q": url, "engine": "google_scholar", "api_key": self.serp_api_key}
        try:
            response = requests.get("https://serpapi.com/search", params=params)
            data = response.json()
            return min(len(data.get("organic_results", [])) * 10, 100)  # Normalize the score
        except:
            return 0  # Return 0 if there's an issue with the citation check

    def detect_content_bias(self, content: str) -> int:
        """ Assesses the potential bias of the content through sentiment analysis. """
        if not content:
            return 50
        sentiment_result = self.model_sentiment(content[:512])[0]
        return 100 if sentiment_result["label"] == "POSITIVE" else 50 if sentiment_result["label"] == "NEUTRAL" else 30

    def convert_to_star_rating(self, score: float) -> tuple:
        """ Converts a numerical score (0-100) into a 1-5 star rating. """
        stars = max(1, min(5, round(score / 20)))  # Normalize the score to a 5-star scale
        return stars, "⭐" * stars

    def generate_feedback(self, domain_trust, similarity_score, fact_check_score, bias_score, citation_score, overall_score) -> str:
        """ Provides a human-readable explanation for the credibility score. """
        reasons = []
        if domain_trust < 50:
            reasons.append("The domain authority of the source is low.")
        if similarity_score < 50:
            reasons.append("The content is not very relevant to your query.")
        if fact_check_score < 50:
            reasons.append("The content has limited fact-checking validation.")
        if bias_score < 50:
            reasons.append("Potential bias has been detected in the content.")
        if citation_score < 30:
            reasons.append("Few citations found for this content.")

        return " ".join(reasons) if reasons else "This source is credible and highly relevant."

    def evaluate_url_credibility(self, user_query: str, url: str) -> dict:
        """ Main function to evaluate the credibility of a webpage. """
        content = self.extract_content(url)

        domain_trust = self.evaluate_domain_authority(url, content)
        similarity_score = self.calculate_semantic_similarity(user_query, content)
        fact_check_score = self.perform_fact_check(content)
        bias_score = self.detect_content_bias(content)
        citation_score = self.analyze_citations(url)

        overall_score = (
            (0.3 * domain_trust) +
            (0.3 * similarity_score) +
            (0.2 * fact_check_score) +
            (0.1 * bias_score) +
            (0.1 * citation_score)
        )

        stars, icon = self.convert_to_star_rating(overall_score)
        feedback = self.generate_feedback(domain_trust, similarity_score, fact_check_score, bias_score, citation_score, overall_score)

        return {
            "raw_scores": {
                "Domain Trust": domain_trust,
                "Content Relevance": similarity_score,
                "Fact-Check Score": fact_check_score,
                "Bias Score": bias_score,
                "Citation Score": citation_score,
                "Overall Credibility Score": overall_score
            },
            "stars": {
                "score": stars,
                "icon": icon
            },
            "feedback": feedback
        }

