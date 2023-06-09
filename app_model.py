from flask import Flask, request, jsonify
import sqlite3
import os
import pickle
import requests
import numpy as np
import pandas as pd
from flask_cors import CORS, cross_origin

app=Flask(__name__)

#método para permitir llamadas

CORS(app, support_credentials=True)

@app.route('/api/test', methods=['POST', 'GET','OPTIONS'])
@cross_origin(supports_credentials=True)
def index():
    if(request.method=='POST'):
     some_json=request.get_json()
     return jsonify({"key":some_json})
    else:
        return jsonify({"GET":"GET"})

#establecer config
@app.after_request
def add_headers(response):
    response.headers.add('Content-Type', 'application/json')
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'PUT, GET, POST, DELETE, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Expose-Headers', 'Content-Type,Content-Length,Authorization,X-Pagination')
    return response

#llamar al archivo
os.chdir(os.path.dirname(__file__))
#debug
app.config['DEBUG'] = True

#1 - endpoint para bienvenida
@app.route("/", methods=['GET'])
def hello():
    return "¡Bienvenida a mi API de predicción de soledad!"


#2 - endpoint para mostrar tabla

@app.route('/datos', methods=['GET'])
def get_users():
    conn = sqlite3.connect('dbdesafio.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM datos')
    users = cursor.fetchall()
    conn.close()
    return jsonify(users)

#3 - endpoint para predecir con los datos que tengo

#cargamos modelo
model = pickle.load(open('best_model_clasif_multi.pkl', 'rb'))

#creamos ruta
@app.route('/predict', methods=['GET'])
def predict():
    #conexión db
    conn = sqlite3.connect('dbdesafio.db')
    cursor = conn.cursor()
    
    #obtener valores
    cursor.execute('SELECT * FROM datos')
    data = cursor.fetchall()
    
    #cerrar conexión
    conn.close()

    #lista vacía para añadir las predicciones
    predictions = []

    for row in data:
        #obtener valores de cada columna de mi db
        edad = row[0]
        estado_civil = row[1]
        sexo = row[2]
        nivel_estudios = row[3]
        psicofarmacos = row[4]
        vive_solo = row[5]
        hijos = row[6]
        ascensor = row[7]
        act_fisica = row[8]
        lim_fisica = row[9]
        estado_animo = row[10]
        satisfaccion_vida = row[11]
        ingresos_economicos = row[12]
        red_apoyo_familiar = row[13]
        cohesion_social = row[14]
        municipio_accesible = row[15]
        municipio_rec_social = row[16]
        municipio_rec_ocio = row[17]

        #predicciones con los atributos entrenados
        prediction = model.predict([[edad, estado_civil, sexo, nivel_estudios, psicofarmacos, vive_solo,
                                     hijos, ascensor, act_fisica, lim_fisica, estado_animo, satisfaccion_vida,
                                     ingresos_economicos, red_apoyo_familiar, cohesion_social, municipio_accesible,
                                     municipio_rec_social, municipio_rec_ocio]])

        #añadir a la lista vacía
        predictions.append(prediction[0])

    #calcula y cuenta las predicciones por cada nivel
    level_counts = {
        'Nivel 0': predictions.count(0),
        'Nivel 1': predictions.count(1),
        'Nivel 2': predictions.count(2)
    }

    #devuelve el json
    return jsonify(level_counts)

#4 - predicciones introduciendo parámetros manualmente

@app.route('/userpred', methods=['GET'])
def userpred():
    model = pickle.load(open('best_model_clasif_multi.pkl','rb'))
    
    edad = request.args.get('edad', None)
    estado_civil = request.args.get('estado_civil', None)
    sexo = request.args.get('sexo', None)
    nivel_estudios = request.args.get('nivel_estudios', None)
    psicofarmacos = request.args.get('psicofarmacos', None)
    vive_solo = request.args.get('vive_solo', None)
    hijos = request.args.get('hijos', None)
    ascensor = request.args.get('ascensor', None)
    act_fisica = request.args.get('act_fisica', None)
    lim_fisica = request.args.get('lim_fisica', None)
    estado_animo = request.args.get('estado_animo', None)
    satisfaccion_vida = request.args.get('satisfaccion_vida', None)
    ingresos_economicos = request.args.get('ingresos_economicos', None)
    red_apoyo_familiar = request.args.get('red_apoyo_familiar', None)
    cohesion_social = request.args.get('cohesion_social', None)
    municipio_accesible = request.args.get('municipio_accesible', None)
    municipio_rec_social = request.args.get('municipio_rec_social', None)
    municipio_rec_ocio = request.args.get('municipio_rec_ocio', None)

    
    if (
        edad is None or estado_civil is None or sexo is None or nivel_estudios is None or
        psicofarmacos is None or vive_solo is None or hijos is None or ascensor is None or
        act_fisica is None or lim_fisica is None or estado_animo is None or
        satisfaccion_vida is None or ingresos_economicos is None or red_apoyo_familiar is None or
        cohesion_social is None or municipio_accesible is None or municipio_rec_social is None or
        municipio_rec_ocio is None
    ):
        return "Faltan valores por especificar"

    else:
        #guardo en variable y convierto a int
        datos = np.array([
            int(edad), estado_civil, sexo, nivel_estudios, psicofarmacos, vive_solo, hijos,
            ascensor, act_fisica, lim_fisica, estado_animo, satisfaccion_vida, ingresos_economicos,
            red_apoyo_familiar, cohesion_social, municipio_accesible, municipio_rec_social,
            municipio_rec_ocio
        ])

        mapeo_riesgo = {
            0: "bajo",
            1: "medio",
            2: "alto"
        }

        prediction = model.predict([datos])

        nivel_riesgo = mapeo_riesgo[prediction[0]]

        # Retorna la respuesta con el nivel de riesgo en palabras
        return "El riesgo de tendencia hacia una soledad no deseada del individuo es: " + nivel_riesgo


if __name__ == '__main__':
    app.run(debug=True)