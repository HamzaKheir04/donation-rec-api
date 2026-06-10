from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# ============================================
# Load Model Files
# ============================================
print("Loading model...")

user_factors = np.load('user_factors.npy')
item_factors = np.load('item_factors.npy')

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

user_encoder          = encoders['user_encoder']
product_encoder       = encoders['product_encoder']
category_encoder      = encoders['category_encoder']
user_category_profile = encoders['user_category_profile']

df         = pd.read_csv('smart_filtered.csv')
df['user_idx']    = user_encoder.transform(df['User_ID'])
df['product_idx'] = product_encoder.transform(df['Product_ID'])
n_products = item_factors.shape[0]

print("✅ Model loaded!")

# ============================================
# Recommendation Function
# ============================================
def get_recommendations(user_id, n=5):
    try:
        user_idx    = user_encoder.transform([user_id])[0]
        user_vector = user_factors[user_idx]
        user_norm   = user_vector / (np.linalg.norm(user_vector) + 1e-10)
        item_norms  = item_factors / (np.linalg.norm(item_factors, axis=1, keepdims=True) + 1e-10)
        svd_scores  = user_norm @ item_norms.T

        user_cats = user_category_profile[
            user_category_profile['User_ID'] == user_id
        ].sort_values('Category_Score', ascending=False)['Category_ID'].values

        cat_scores = np.zeros(n_products)
        for cat in user_cats[:3]:
            cat_products = df[df['Category_ID'] == cat]['product_idx'].values
            cat_scores[cat_products] += 1

        if cat_scores.max() > 0:
            cat_scores = cat_scores / cat_scores.max()

        hybrid_scores = (0.6 * svd_scores) + (0.4 * cat_scores)

        seen = df[df['User_ID'] == user_id]['product_idx'].values
        hybrid_scores[seen] = -1

        top_indices  = hybrid_scores.argsort()[::-1][:n]
        top_products = product_encoder.inverse_transform(top_indices)
        top_cats     = [
            df[df['product_idx'] == idx]['Category_ID'].iloc[0]
            if len(df[df['product_idx'] == idx]) > 0 else None
            for idx in top_indices
        ]
        top_scores = hybrid_scores[top_indices]

        return {
            'user_id'        : int(user_id),
            'recommendations': [
                {
                    'product_id' : int(top_products[i]),
                    'category_id': int(top_cats[i]) if top_cats[i] else None,
                    'score'      : round(float(top_scores[i]), 4)
                }
                for i in range(len(top_products))
            ]
        }

    except Exception as e:
        return None

# ============================================
# Routes
# ============================================
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': '✅ Recommendation API is running!'})

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')

    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400

    try:
        user_id = int(user_id)
    except:
        return jsonify({'error': 'user_id must be a number'}), 400

    result = get_recommendations(user_id, n=5)

    if result is None:
        return jsonify({'error': 'User not found in training data'}), 404

    return jsonify(result)

# ============================================
# Run
# ============================================
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)