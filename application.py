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
    base_url = os.environ.get('BASE_URL', 'http://127.0.0.1:5000')
    return render_template('index.html', base_url=base_url)


@application.route('/idea/<idea_id>')
def view_idea(idea_id):
    """Display a single idea by ID"""
    try:
        # Get idea from repository
        if idea_id not in system.repository.graph.nodes:
            return "Idea not found", 404
        
        idea_data = system.repository.graph.nodes[idea_id]
        
        # Get evaluations
        evals = system.repository.get_evaluations(idea_id)
        evaluation = evals[0] if evals else None
        
        return render_template('idea_detail.html', 
                             idea_id=idea_id,
                             idea=idea_data,
                             evaluation=evaluation)
    except Exception as e:
        return f"Error loading idea: {str(e)}", 500


@application.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Receive and store human feedback for an idea"""
    try:
        data = request.json
        print(f"Received feedback: {data}")
        
        # Validate input
        if not data or 'idea_id' not in data or 'rating' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        idea_id = data['idea_id']
        rating = int(data['rating'])
        comment = data.get('comment', '')
        # Use server time if not provided, or simply use server time to be safe
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        # Validate rating
        if rating < 1 or rating > 5:
            return jsonify({'error': 'Rating must be between 1 and 5'}), 400
        
        # Check if idea exists
        if idea_id not in system.repository.graph.nodes:
            return jsonify({'error': 'Idea not found'}), 404
        
        # Store feedback
        system.repository.add_human_feedback(idea_id, rating, comment, timestamp)
        
        return jsonify({'success': True, 'message': 'Feedback recorded'}), 200
        
    except Exception as e:
        print(f"Error processing feedback: {e}")
        return jsonify({'error': str(e)}), 500


@application.route('/api/process', methods=['POST'])
def process():
    data = request.json
    problem = data.get('problem')
    mode = data.get('mode', 'mix')
    repo_only = data.get('repoOnly', False)
    
    # Backward compat
    if repo_only:
        mode = 'repository'
        
    limit = int(data.get('limit', 5))
    ideas_per_persona = int(data.get('ideasPerPersona', 1))
    
    if not problem:
        return jsonify({"error": "Problem statement required"}), 400
    try:
        # Capture logs for this request
        log_buffer = setup_logging(capture_logs=True)
        
        results = system.process_problem(problem, mode=mode, limit=limit, ideas_per_persona=ideas_per_persona)
        return jsonify({
            "results": results,
            "logs": log_buffer
        })
    except Exception as e:
        print(f"Error processing: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # application.run(debug=True, host='0.0.0.0', port=5000)
    application.run(port=5000)
