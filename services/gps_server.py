# from flask import Flask, request, jsonify
# from services.gps_state import update_gps

# app = Flask(__name__)

# @app.route("/gps", methods=["POST"])
# def receive_gps():
#     data = request.get_json()
#     if not data or "lat" not in data or "lon" not in data:
#         return jsonify({"error": "Invalid GPS data"}), 400

#     update_gps(data)
#     print(f"[GPS] 위치 수신됨: 위도={data['lat']}, 경도={data['lon']}")
#     return jsonify({"status": "ok"})

# def start_gps_server():
#     app.run(host="0.0.0.0", port=8080)
