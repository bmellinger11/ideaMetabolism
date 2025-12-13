import os
from flask import Flask, render_template, request, jsonify
from idea_metabolism import IdeaMetabolismSystem

app = Flask(__name__)

# Initialize system globally
# In production, this might need better management, but for POC it's fine.
system = IdeaMetabolismSystem(llm_provider="anthropic")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process():
    data = request.json
    problem = data.get('problem')
    repo_only = data.get('repoOnly', False)
    limit = int(data.get('limit', 5))
    
    if not problem:
        return jsonify({"error": "Problem statement required"}), 400
    
    try:
        results = system.process_problem(problem, repo_only=repo_only, limit=limit)
        return jsonify({"results": results})
    except Exception as e:
        print(f"Error processing: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
