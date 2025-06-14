"""
Contains all Flask route definitions for document compliance analysis.
Each route is documented below.
"""

from flask import request, jsonify
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer

from utils import (
    initialize_system, documents, policy_metadata, vector_store,
    search_vector_store, analyze_compliance, extract_text_from_file,
    segment_text_into_clauses, filter_relevant_clauses, compare_clauses,
    check_coverage_from_standard
)
from collections import defaultdict
import os

def register_routes(app):
    @app.route('/initialize', methods=['POST'])
    def initialize():
        """
        Initializes the system with a list of policy `.docx` files.

        Expects:
            - Multipart form data with key `files`: list of `.docx` files.

        Returns:
            - JSON: Success or error message and number of chunks loaded.
        """
        if 'files' not in request.files:
            return jsonify({"error": "No files uploaded"}), 400

        files = request.files.getlist("files")
        saved_paths = []

        for file in files:
            if file.filename.endswith(".docx"):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                saved_paths.append(filepath)

        try:
            initialize_system(saved_paths)
            return jsonify({"status": "success", "message": f"Loaded {len(documents)} policy chunks"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/analyze', methods=['POST'])
    def analyze():
        """
        Analyzes how well existing policies cover a given use case.

        Expects:
            - JSON with key `use_case`: a string describing the use case.

        Returns:
            - JSON: Gemini-based analysis, KPI scores, and source policy mapping.
        """
        try:
            if not vector_store:
                return jsonify({"error": "System not initialized with policies"}), 400
            if 'use_case' not in request.json:
                return jsonify({"error": "use_case parameter required"}), 400

            use_case = request.json['use_case']
            results = search_vector_store(use_case, k=5)
            relevant_chunks = [documents[idx] for idx, _ in results]
            relevant_metadata = [policy_metadata[idx] for idx, _ in results]

            analysis = analyze_compliance(use_case, relevant_chunks)
            analysis["source_policies"] = defaultdict(list)
            for meta in relevant_metadata:
                analysis["source_policies"][meta["policy"]].append({
                    "chunk_id": meta["chunk_id"],
                    "source_file": meta["source"]
                })
            analysis["source_policies"] = dict(analysis["source_policies"])

            return jsonify(analysis)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/check_compliance', methods=['POST'])
    def check_compliance():
        """
        Checks compliance of a user-provided policy against one or more standard files.

        Expects:
            - Multipart form data:
                - `policy_file`: the `.docx` policy file to check.
                - `standard_files`: one or more `.pdf` or `.docx` standard documents.
                - `threshold` (optional): float similarity threshold (default = 0.75)

        Returns:
            - JSON:
                - compliance_score: percentage of standard clauses matched
                - matched_clauses: list of matched clause pairs with similarity
                - missing_clauses: list of unmatched standard clauses
                - user_clauses_count, standard_clauses_count
        """
        try:
            if 'standard_files' not in request.files or 'policy_file' not in request.files:
                return jsonify({"error": "standard_files and policy_file are required"}), 400

            standard_files = request.files.getlist('standard_files')
            policy_file = request.files['policy_file']

            policy_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(policy_file.filename))
            policy_file.save(policy_path)
            policy_text = extract_text_from_file(policy_path)
            policy_clauses_raw = segment_text_into_clauses(policy_text)

            model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            policy_clauses = filter_relevant_clauses(policy_clauses_raw, model, threshold=0.4)

            combined_standard_clauses = []
            for std_file in standard_files:
                std_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(std_file.filename))
                std_file.save(std_path)
                std_text = extract_text_from_file(std_path)
                std_clauses_raw = segment_text_into_clauses(std_text)
                std_clauses = filter_relevant_clauses(std_clauses_raw, model, threshold=0.4)
                combined_standard_clauses.extend(std_clauses)
                os.remove(std_path)

            os.remove(policy_path)

            threshold = float(request.form.get('threshold', 0.75))
            matches = compare_clauses(policy_clauses, combined_standard_clauses, model, threshold)
            matched, missing = check_coverage_from_standard(combined_standard_clauses, policy_clauses, model, threshold)
            compliance_score = len(matched) / len(combined_standard_clauses) if combined_standard_clauses else 0

            return jsonify({
                "status": "success",
                "compliance_score": compliance_score,
                "matched_clauses": matched,
                "missing_clauses": missing,
                "user_clauses_count": len(policy_clauses),
                "standard_clauses_count": len(combined_standard_clauses)
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/health', methods=['GET'])
    def health():
        """
        Health check endpoint.

        Returns:
            - JSON: Service status and whether policies have been initialized.
        """
        return jsonify({"status": "ok", "initialized": vector_store is not None})