from flask import Flask , jsonify , request , abort
import pymongo
from chain import Chain
from block import Block
from pprint import pprint
import json

chain = Chain()
app = Flask(__name__)


myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["GenBook"]
blockchainDB = mydb["BlockchainDB"]


@app.route('/api/v1/genbook' , methods = ['GET'])
def homepage():
    response = {'message' : 'blockchain is working' ,
                'status_code' : 'success'}
    return jsonify(response) , 200


@app.route('/api/v1/blockchain' , methods = ['GET'])
def getBlockchain():
    chain = []
    for x in blockchainDB.find({} , {"_id": 0}):
        chain.append(x)
    response = {'message' : 'blockchain is working' ,
                'data' : chain}
    return jsonify(response) , 200

def readDataFromJSON():
    with open('dataPrevBlock.json') as f:
        data = json.load(f)
    
    return data

def writeDataToJSON(prevProof , prevHash , lengthChain):
    with open('dataPrevBlock.json' , 'w') as fw:
        json.dump({
            "prevProof":prevProof,
            "prevHash":prevHash,
            "lengthChain":lengthChain
        } , fw)    

@app.route('/api/v1/blockchain' , methods = ['POST'])
def addNewBlock():

    data = readDataFromJSON()
    newBlock = Block(data["prevProof"] , data["prevHash"] , data["lengthChain"])
    if not request.json or not 'transaction' in request.json:
        abort(400)

    for x in request.json['transaction']:
        newBlock.addTransaction(x)
    
    newBlock.generateHash()
    block = {
        "hash" : newBlock.hash,
        "index" : data["lengthChain"] + 1,
        "prevProof" : data["prevProof"],
        "previousHash" : data["prevHash"],
        "proof" : newBlock.proof,
        "timestamp" : newBlock.timestamp,
        "transaction" : request.json['transaction']
    }
    blockchainDB.insert_one(block)

    prevHash = newBlock.hash
    prevProof = newBlock.proof
    lengthChain = data["lengthChain"] + 1

    writeDataToJSON(prevProof , prevHash , lengthChain)

    return jsonify(
        {'result':'success'}) , 200


app.run(host='0.0.0.0' , port=5000)