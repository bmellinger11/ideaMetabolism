import os
from flask import Flask, render_template, request, jsonify
from idea_metabolism import IdeaMetabolismSystem, setup_logging

application = Flask(__name__)

# Initialize system globally
# In production, this might need better management, but for POC it's fine.
system = IdeaMetabolismSystem(llm_provider="anthropic")
# OPTIONAL: system = IdeaMetabolismSystem(llm_provider="openai")
# OPTIONAL: system = IdeaMetabolismSystem(llm_provider="gemini")


@application.route('/')
def home():
    return render_template('index.html')


@application.route('/api/process', methods=['POST'])
def process():
    data = request.json
    problem = data.get('problem')
    repo_only = data.get('repoOnly', False)
    limit = int(data.get('limit', 5))
    ideas_per_persona = int(data.get('ideasPerPersona', 1))
    
    if not problem:
        return jsonify({"error": "Problem statement required"}), 400
    
    try:
        # Capture logs for this request
        log_buffer = setup_logging(capture_logs=True)
        
        results = system.process_problem(problem, repo_only=repo_only, limit=limit, ideas_per_persona=ideas_per_persona)
        return jsonify({
            "results": results,
            "logs": log_buffer
        })
    except Exception as e:
        print(f"Error processing: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0', port=5000)
