# Django Imports
from django.shortcuts import render, redirect
from django.contrib import messages
from django.utils.decorators import method_decorator
from django.contrib.auth.models import Group
from django.views import View

# Internal Imports
from . import mongodb as mdb
from .forms import CreateUserForm
from admin_classifier.views import ConcreteSubject, ConcreteLearner

# Python Package Imports
import pickle
import numpy as np

# Create your views here.
db_data = mdb.access()


# Home and Login Module Starts
class Home(View):

    template_name = 'user_classifier/home.html'
    context = {}

    def get(self, request):
        username = str(request.user)
        learner = ConcreteLearner()
        mongo_data = learner.update()
        len_update_messages_list = len(mongo_data['update_message_list'])
        temp = "\n".join(mongo_data['update_message_list'])
        update_messages_list = [temp]
        observer_list = mongo_data['observer_list']
        self.context = {'username': str(username),
                        'observer_list': observer_list,
                        'update_messages_list': update_messages_list,
                        'len_update_messages_list': len_update_messages_list}
        return render(request, self.template_name, self.context)

    def post(self, request):
        username = str(request.user)
        learner = ConcreteLearner()
        mongo_data = learner.update()
        subject = ConcreteSubject(mongo_data["observer_list"], mongo_data["update_message_list"])
        if request.POST.get('subscribe') == "Subscribe Updates":
            alert_message = subject.subscribe(username)
        if request.POST.get('unsubscribe') == "Unsubscribe Updates":
            alert_message = subject.unsubscribe(username)
        mongo_data = learner.update()
        len_update_messages_list = len(mongo_data['update_message_list'])
        temp = "\n".join(mongo_data['update_message_list'])
        update_messages_list = [temp]
        observer_list = mongo_data['observer_list']
        self.context = {'username': username,
                        'observer_list': observer_list,
                        'update_messages_list': update_messages_list,
                        'len_update_messages_list': len_update_messages_list}
        return render(request, self.template_name, self.context)

# Simple Factory Pattern Starts
class Algorithm(View):

    def get(self, request):
        pass

    def post(self, request):
        pass


class Classification(Algorithm):

    template_name = 'user_classifier/classification.html'
    message = ""
    submit_button = None

    def get(self, request):

        data = mdb.find(db_data, "KNN")
        context = {'algo_desc': data['algo_desc'], 'ds_desc': data['ds_desc'],
                   'training_features': data['training_features']}

        return render(request, self.template_name, context)

    def post(self, request):

        data = mdb.find(db_data, "KNN")
        graph_image = data['graph_image']
        if data['upload_method'] == 'pkl':
            classifier = pickle.loads(pickle.loads(data['pkl_data']).read())
        else:
            classifier = pickle.loads(data['pkl_data'])
        if 'submit' in request.POST:
            self.submit_button = request.POST.get("submit")
            output_message = data['label_notes']
            user_inputs = [np.array(request.POST.getlist('user_inputs')).astype(np.float64)]
            sc = pickle.loads(data['scaling_obj'])

            try:
                user_inputs = sc.transform(user_inputs)
                preds = classifier.predict(user_inputs)
                self.message = output_message[str(preds[0])]
                if data['upload_method'] == 'csv':
                    accuracy = data['testing_accuracy']
                    f1_score = data['f1_score']
                    extra = " (" + str(accuracy) + "% accuracy and " + str(f1_score) + "% F1-Score)"
                    self.message = self.message + extra
            except:
                self.message = "Unexpected error while predicting the output"

        context = {'algo_desc': data['algo_desc'], 'ds_desc': data['ds_desc'],
                   'training_features': data['training_features'], 'graph_image': graph_image,
                   'message': self.message, 'submitbutton': self.submit_button}

        return render(request, self.template_name, context)


class Regression(Algorithm):

    template_name = 'user_classifier/regression.html'
    message = ""
    submit_button = None

    def get(self, request):

        data = mdb.find(db_data, "MLR")
        context = {'algo_desc': data['algo_desc'], 'ds_desc': data['ds_desc'],
                   'training_features': data['training_features']}

        return render(request, self.template_name, context)

    def post(self, request):

        data = mdb.find(db_data, "MLR")
        graph_image = data['graph_image']
        if data['upload_method'] == 'pkl':
            regressor = pickle.loads(pickle.loads(data['pkl_data']).read())
        else:
            regressor = pickle.loads(data['pkl_data'])
        if 'submit' in request.POST:
            self.submit_button = request.POST.get("submit")
            user_inputs = [np.array(request.POST.getlist('user_inputs')).astype(np.float64)]
            sc = pickle.loads(data['scaling_obj'])

            try:
                user_inputs = sc.transform(user_inputs)
                preds = regressor.predict(user_inputs)
                self.message = "The predicted profit of the startup is " + str(round(preds[0], 2))
                if data['upload_method'] == 'csv':
                    rmse = data['rmse']
                    extra = " (With " + str(rmse) + " Root Mean-Squared Error)"
                    self.message = self.message + extra
            except:
                self.message = "Unexpected error while predicting the output"

        context = {'algo_desc': data['algo_desc'], 'ds_desc': data['ds_desc'],
                   'training_features': data['training_features'], 'graph_image': graph_image,
                   'message': self.message, 'submitbutton': self.submit_button}

        return render(request, self.template_name, context)


class Clustering(Algorithm):

    template_name = 'user_classifier/clustering.html'
    message = ""
    submit_button = None

    def get(self, request):

        data = mdb.find(db_data, "KM")
        context = {'algo_desc': data['algo_desc'], 'ds_desc': data['ds_desc'],
                   'training_features': data['training_features']}

        return render(request, self.template_name, context)

    def post(self, request):

        data = mdb.find(db_data, "KM")
        graph_image = data['graph_image']
        if data['upload_method'] == 'pkl':
            classifier = pickle.loads(pickle.loads(data['pkl_data']).read())
        else:
            classifier = pickle.loads(data['pkl_data'])
        if 'submit' in request.POST:
            self.submit_button = request.POST.get("submit")
            output_message = data['label_notes']
            user_inputs = np.array(request.POST.getlist('user_inputs')).astype(np.float64)
            try:
                preds = classifier.predict([user_inputs])
                self.message = output_message[str(preds[0])]
            except:
                self.message = "Unexpected error while predicting the output"

        context = {'algo_desc': data['algo_desc'], 'ds_desc': data['ds_desc'],
                   'training_features': data['training_features'], 'graph_image': graph_image,
                   'message': self.message, 'submitbutton': self.submit_button}

        return render(request, self.template_name, context)
# Simple Factory Pattern Ends
