# more consumer.py
from flask import Flask, Response
from kafka import KafkaConsumer
#connect to Kafka server and pass the topic we want to consume
consumer = KafkaConsumer('my-topic', group_id='view', bootstrap_servers=['127.0.0.1:9092'])

if not consumer:
   print('ERROR: consumer not connected. Check the consumer IP address and port numbers.')
   sys.exit()

#Continuously listen to the connection and print messages as recieved
app = Flask(__name__)

@app.route('/')
def index():
    # return a multipart response
    return Response(kafkastream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def kafkastream():
 print('kafka stream ')
 i = 0
 for msg in consumer:
  print('msg number: ' + str(i))
  yield (b'--frame\r\n'
         b'Content-Type: image/png\r\n\r\n' + msg.value + b'\r\n\r\n')

if __name__ == '__main__':
 print('main')
 app.run()
#    app.run(host='127.0.0.1', debug=True)


