# ДЗ

В этой домашней работе вам предстоит решить задачу ассоциации изображений методами metric learning. Необходимо обучить нейронную сеть для поиска похожих автомобилей из датасета cars196. Учтите, что для формирования предсказаний на тестовой выборке вы не должны предсказывать напрямую класс объекта (не используем метод прямой классификации). Пример формирования предсказаний можно найти в ноутбуке: [metric_learning.ipynb](metric_learning.ipynb) блок Faiss

В процессе решения нужно решить пункты:
1. Реализовать метрики: Precision@k, Recall@k, mAP 
2. Добавить train аугментации 
3. Обучить модель,
  * При обучении использовать backbone отличный от того, что использовался на семинаре (не ResNet50)
  *  Добавить triplet-loss (помните о грамотном формировании триплетов)
4. Рассчитать метрики из п.1
5. Показать с помощью град Кама, где ошибается модель
