from kivy.uix.screenmanager import Screen
# from kivy.properties import ObjectProperty
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import svds
import json

# Переменная не используется при запросе на получение курсов, убрать требование к авторизации на сервере
token = ""


class ScreenLogin(Screen):
    def auth(self):
        # Получаем логин из текстбокса
        login = self.ids['tbLogin'].text
        # Получаем пароль из текстбокса
        password = self.ids['tbPassword'].text

        # Аутентификация
        access = requests.post('http://localhost:5275/api/auth',
                               headers={'Content-Type': 'application/json', 'Accept': "application/json"},
                               json={"username": login, "password": password},
                               verify=False)

        response = json.loads(access.text)

        if "access_token" in response.keys():
            if response["access_token"] != "" and response["role"] == 'admin':
                print("Доступ получен!")
                token = response["access_token"]
                return True # Преходим на главную страницу
        print("Доступ отклонен.")
        return False # Не переходим на главную страницу


class ScreenMain(Screen):
    def predcition(self):
        # Запрос на получение рейтингов
        request1 = requests.get("http://localhost:5275/api/users/get-ratings",
                                headers={'Accept': "application/json", 'Authorization': "Bearer " + token},
                                verify=False)
        response1 = json.loads(request1.text)

        # Сохранение текущих рейтингов в dataFrame
        df = pd.DataFrame(response1)
        # Запрос на получение курсов
        request2 = requests.get("http://localhost:5275/api/courses",
                                headers={'Accept': "application/json", 'Authorization': "Bearer " + token},
                                verify=False)
        response2 = json.loads(request2.text)
        # Сохранение курсов в dataFrame
        df2 = pd.DataFrame(response2)


        print(df)
        print(df2)
        print()
        # df = pd.read_json("http://localhost:5275/api/users/get-ratings")
        # df2 = pd.read_json("http://localhost:5275/api/courses")

        # df = df.sort_values(by = 'userId')

        # shape возвращает размерность массива в виде кортежа
        # Кол-во пользователей по Id
        n_user = df['userId'].max()
        # Кол-во курсов по Id
        n_items = df2['id'].max()
        print(n_items)

        print(f"Данные:\n{df}")
        print(f"Число пользователей: {n_user}")
        print(f"Число продуктов: {n_items}")

        """ Вывод гистограммы """
        # Ошибка в выводе гистограммы - баг, нужно спуститься до python 3.9
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        sns.distplot(df.rating, color="blue", ax=ax, kde=False)
        ax.set_title("Рейтинг", fontsize=20)
        plt.tight_layout()
        plt.show()

        """ Преобразование данных в матричную форму """
        data_matrix = np.zeros((n_user, n_items))
        print(data_matrix)
        for line in df.itertuples():
            # [user_id][product_id] = rating
            print(f"\t>> {line}")
            data_matrix[line[1] - 1, line[2] - 1] = line[3]
            print(f"\t\t {data_matrix[line[1] - 1, line[2] - 1]}")

        print("Матрица полезности: ")
        print("", end="\t\t\t")
        for i in range(n_items):
            print(f"p{i + 1}", end="\t\t")
        print()
        for i in range(n_user):
            print(f"user{i + 1}.", end="\t\t")
            for j in range(n_items):
                print(f"{data_matrix[i][j]}", end="\t\t")
            print()

        """ Колобаративная фильтрация - User-Based подход
            Найдем похожих пользователей, и спрогнозируем оценки, заменив нули """

        # разряженная матрица - матрица приемущественно из нулей
        # разбиваем ее на произведение ортогональных матриц и диагональной
        # разложив исходную на компоненты, мы можем вновь их переумножить и получить "восстановленную матрицу"

        # Реализация SVD
        u, s, vt = svds(data_matrix, k=6)
        s_diag_matrix = np.diag(
            s)  # s - диагональная матрица, у нее везде 0, кроме главной диагонали (там находятся сигмы)
        s_diag_matrix = np.round(s_diag_matrix, 1)  # округляем до десятых (1 знак после запятой)

        # Матрица с предсказанными рейтингами
        predict_matrix = np.dot(np.dot(u, s_diag_matrix), vt)  # np.dot - умножение матриц
        predict_matrix = np.round(predict_matrix, 1)

        print('Предсказываем рейтинги-заменяем ими нули- для матрицы данных:')

        print("", end="\t\t\t")
        for i in range(n_items):
            print(f"p{i + 1}", end="\t\t")
        print()
        for i in range(n_user):
            print(f"user{i + 1}.", end="\t\t")
            for j in range(n_items):
                print(f"{predict_matrix[i][j]}", end="\t\t")
            print()
        # что такое метрика?

        # Выводим предсказания в графике
        predict_matrix_T = predict_matrix.T  # Транспонируем
        # Вывод матрицы с прогнозами в красивом виде (транспониуем)
        fig, ax = plt.subplots()
        M = predict_matrix_T[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]][:,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
        ax.matshow(M, cmap=plt.cm.Reds)
        for i in range(19):
            for j in range(15):
                c = predict_matrix_T[j, i]
                ax.text(i, j, str(c), va='center', ha='center')
        plt.xlabel('Пользователи')
        plt.ylabel('Курсы')
        plt.title('Фрагмент предсказанных рейтингов курсов')
        plt.show()

        def rmse(prediction, ground_truth):
            prediction = prediction[ground_truth.nonzero()].flatten()
            ground_truth = ground_truth[ground_truth.nonzero()].flatten()
            return sqrt(mean_squared_error(prediction, ground_truth))

        print('User-based CF MSE: ' + str(rmse(predict_matrix, data_matrix)))

        def CheckForSend(userId, courseId):
            finded = df.loc[(df['userId'] == userId) & (df['courseId'] == courseId)]

            if (finded.empty) | (finded['rating'] == 0).any():
                return True
            else:
                return False



        def JsonGenerator(p_matrix):
            data = []
            for i in range(len(p_matrix)):
                for j in range(len(p_matrix[i])):
                    if CheckForSend(i + 1, j + 1):
                        data.append(GenRow(i + 1, j + 1, p_matrix[i][j]))
            return data

        def GenRow(userId, courseId, rating):
            return {'userId': userId,
                    'courseId': courseId,
                    'rating': rating}

        print("\n-------------------------------------------------------------\n")
        print(df)
        print(predict_matrix)
        print("СМОТРИ ВЫШЕ")

        '''for line in df.itertuples():
            # [user_id][product_id] = rating
            print(f"\t>> {line}")
            predict_matrix[line[1] - 1, line[2] - 1] = line[3]
            print(f"\t\t {predict_matrix[line[1] - 1, line[2] - 1]}")'''

        pData = JsonGenerator(predict_matrix)
        print(pData)

        # with open(f'data.json', 'w') as file:
        #    json.dump(pData, file)

        # /api/uploadfiles/recsys
        # files = {'data.json': open('data.json', 'rb')}

        access = requests.post('http://localhost:5275/api/recsystem/upload-predicts',
                               headers={'Content-Type': 'application/json', 'Accept': "application/json",
                                        'Authorization': "Bearer " + token},
                               json=pData,
                               verify=False)




