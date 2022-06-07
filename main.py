import requests
from kivy.config import Config

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, NoTransition
from ScreenLogReg import ScreenLogin, ScreenMain


Config.set('graphics', 'resizable', '0')
Config.set('graphics', 'width', '400')
Config.set('graphics', 'height', '710')


class RobRecSysApp(App):
    def build(self):
        """ Метод построения приложения """

        sm = ScreenManager(transition=NoTransition())   # менеджер окон
        sl = ScreenLogin(name='screen_login')   # окно авторизации
        smain = ScreenMain(name='screen_main')  # домашняя страница приложения

        # Добавление окон в менеджер
        sm.add_widget(sl)
        sm.add_widget(smain)

        return sm

    def show_alert_dialog(self):
        print("кнопка нажата")
    # root.manager.current = 'screen_main' - переключ на экран kivy



#self.manager.screeens[номер экрана в screenmanager].ids.<id виджета>.<его атрибут>


if __name__ == '__main__':
    RobRecSysApp().run()

