from flask import Flask, render_template_string, request, jsonify
import cv2
import numpy as np
import time
import paho.mqtt.client as mqtt
import json
import base64

app = Flask(__name__)
# ===== CONFIG =====
BROKER = "10.42.0.235"
TOPIC = "edgeai/face_detection"
mqtt_data = {"face_count":0,"fps":0,"latency":0}

# ===== MODEL =====
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
)

# ===== MQTT =====
def on_connect(client,userdata,flags,rc):
    client.subscribe(TOPIC)
    
def on_message(client,userdata,msg):
    global mqtt_data
    data=json.loads(msg.payload.decode())
    now=time.time()
    ts=data.get("timestamp",now)
    mqtt_data["face_count"]=data.get("face_count",0)
    mqtt_data["fps"]=data.get("fps",0)
    mqtt_data["latency"]=round((now-ts)*1000,2)
    
client=mqtt.Client()
client.on_connect=on_connect
client.on_message=on_message
client.connect(BROKER,1883,60)
client.loop_start()

# ===== NMS =====
def nms(boxes, thresh=0.2):
    if len(boxes)==0:
        return []
    boxes=np.array(boxes)
    x1=boxes[:,0]; y1=boxes[:,1]
    x2=boxes[:,2]; y2=boxes[:,3]
    area=(x2-x1+1)*(y2-y1+1)
    idxs=np.argsort(y2)
    pick=[]
    while len(idxs)>0:
        i=idxs[-1]
        pick.append(i)
        xx1=np.maximum(x1[i],x1[idxs[:-1]])
        yy1=np.maximum(y1[i],y1[idxs[:-1]])
        xx2=np.minimum(x2[i],x2[idxs[:-1]])
        yy2=np.minimum(y2[i],y2[idxs[:-1]])
        w=np.maximum(0,xx2-xx1+1)
        h=np.maximum(0,yy2-yy1+1)
        overlap=(w*h)/area[idxs[:-1]]
        idxs=np.delete(idxs,
            np.concatenate(([len(idxs)-1],
            np.where(overlap>thresh)[0])))
    return boxes[pick]
    
# ===== HTML =====
HTML="""
<html>
<head>
<style>
body{font-family:Arial;background:#f4f6f9;text-align:center}
.nav{background:#111;padding:15px}
.nav a{color:white;margin:20px;text-decoration:none;font-weight:bold}
.card{background:white;padding:20px;margin:20px auto;width:700px;
border-radius:12px;box-shadow:0 5px 15px rgba(0,0,0,0.2)}
img{border-radius:10px;border:4px solid #333}
</style>
<script>
function update(){
 fetch("/data")
 .then(r=>r.json())
 .then(d=>{
  document.getElementById("fc").innerText=d.face_count
  document.getElementById("fps").innerText=d.fps
  document.getElementById("lat").innerText=d.latency
 })
}
setInterval(update,1000)
</script>
</head>
<body>
<div class="nav">
<a href="/?mode=realtime">Realtime</a>
<a href="/?mode=static">Static</a>
</div>
<div class="card">
<h2>MQTT DATA</h2>
Face: <span id="fc">0</span><br>
FPS: <span id="fps">0</span><br>
Latency: <span id="lat">0</span> ms
</div>
{% if mode=="realtime" %}
<div class="card">
<h2>Realtime Camera</h2>
<img src="http://10.42.0.235:8000/video_feed" width="650">
</div>
{% endif %}
{% if mode=="static" %}
<div class="card">
<h2>Upload Image</h2>
<form method="POST" enctype="multipart/form-data">
<input type="file" name="file">
<br><br>
<input type="submit" value="Detect">
</form>
</div>
{% endif %}
{% if image %}
<div class="card">
<h2>Result</h2>
<p><b>Face detected:</b> {{count}}</p>
<img src="{{image}}" width="650">
</div>
{% endif %}
</body>
</html>
"""

# ===== ROUTES =====
@app.route("/data")
def data():
    return jsonify(mqtt_data)
    
@app.route("/",methods=["GET","POST"])
def index():
    mode=request.args.get("mode","realtime")
    image_data=None
    face_count=0
    if request.method=="POST":
        file=request.files.get("file")
        if file:
            img=cv2.imdecode(
                np.frombuffer(file.read(),np.uint8),
                cv2.IMREAD_COLOR
            )
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # ===== DETECT (STRICT) =====
            faces=face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=7,
                minSize=(80,80)
            )
            # ===== FILTER =====
            boxes=[]
            for (x,y,w,h) in faces:
                if w*h < 5000:
                    continue
                boxes.append([x,y,x+w,y+h])
            # ===== NMS =====
            boxes=nms(boxes,0.2)
            # ===== KEEP BEST =====
            if len(boxes)>0:
                boxes=sorted(
                    boxes,
                    key=lambda b:(b[2]-b[0])*(b[3]-b[1]),
                    reverse=True
                )[:1]
            face_count=len(boxes)
            # ===== DRAW =====
            for (x1,y1,x2,y2) in boxes:
                w=x2-x1; h=y2-y1
                conf=min(1.0,(w*h)/(150*150))
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),4)
                cv2.putText(img,
                    str(round(conf*100,1))+"%",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,(0,255,0),3)
            _,buffer=cv2.imencode(".jpg",img)
            image_data="data:image/jpeg;base64,"+base64.b64encode(buffer).decode()
            
    return render_template_string(
        HTML,
        mode=mode,
        image=image_data,
        count=face_count
    )
    
# ===== RUN =====
if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)
