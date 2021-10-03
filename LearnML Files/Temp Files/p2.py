from pymongo import MongoClient

client = MongoClient("mongodb+srv://user_1:USER_1@cluster0.0oqke.mongodb.net/<dbname>?retryWrites=true&w=majority")
db = client.get_database('learnml_db')
db_data = db.algorithms

mongo_data = {"update_message_list": ["Welcome To Learn ML"]}
db_data.update_one({'name': 'OP'}, {'$set': mongo_data})
