[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Daemon2017_yImputer&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Daemon2017_yImputer)
# yImputer

## Как пользоваться?
* скачайте Postman: https://www.postman.com/downloads/?utm_source=postman-home
* перейдите по ссылке https://raw.githubusercontent.com/Daemon2017/yImputer/master/yImputer.postman_collection.json и нажмите Ctrl+S, чтобы сохранить JSON-файл с коллекцией запросов для Postman;
* установите Postman;
* запустите Postman;
* нажмите Ctrl+O, чтобы импортировать JSON-коллекцию в Postman;
* в меню слева появится коллекция yImputer с 4 запросами в ней.

## Виды взаимодействий
### JSON Endpoint
Примнимает на вход массив JSON'ов (от 1 до сотен строк), содержащий от 1 до 111 маркеров. Если какой-либо STR неизвестен - уберите его пару **"ключ": значение** из запроса. Пример запроса есть в Postman-коллекции. Ответ будет содержать JSON с предсказанными 111 STR в стандартном порядке FTDNA.

### CSV Endpoint
Примнимает на вход массив CSV (от 1 до сотен строк), содержащий от 1 до 111 маркеров. Если какой-либо STR неизвестен - либо уберите его столбец из запроса, либо оставьте пустой ячейку со значением. Порядок STR не имеет значения. CSV-файл может иметь любой разделитель, который можно задать с помощью URL-аргумента, например: "**?sep=,**". Пример CSV-файла, использующего запятую в качестве разделителя:
```
DYS393,DYS390,DYS19,DYS391,DYS385a,DYS385b,DYS426,DYS388,DYS439,DYS389I,DYS392,DYS389II,DYS458,DYS459a,DYS459b,DYS455,DYS454,DYS447,DYS437,DYS448,DYS449,DYS464a,DYS464b,DYS464c,DYS464d,DYS460,Y-GATA-H4,YCAIIa,YCAIIb,DYS456,DYS607,DYS576,DYS570,CDYa,CDYb,DYS442,DYS438,DYS531,DYS578,DYF395S1a,DYF395S1b,DYS590,DYS537,DYS641,DYS472,DYF406S1,DYS511,DYS425,DYS413a,DYS413b,DYS557,DYS594,DYS436,DYS490,DYS534,DYS450,DYS444,DYS481,DYS520,DYS446,DYS617,DYS568,DYS487,DYS572,DYS640,DYS492,DYS565,DYS710,DYS485,DYS632,DYS495,DYS540,DYS714,DYS716,DYS717,DYS505,DYS556,DYS549,DYS589,DYS522,DYS494,DYS533,DYS636,DYS575,DYS638,DYS462,DYS452,DYS445,Y-GATA-A10,DYS463,DYS441,Y-GGAAT-1B07,DYS525,DYS712,DYS593,DYS650,DYS532,DYS715,DYS504,DYS513,DYS561,DYS552,DYS726,DYS635,DYS587,DYS643,DYS497,DYS510,DYS434,DYS461,DYS435
13,25,16,10,12,13,12,13,10,13,11,30,15,9,10,11,11,,14,20,33,,,,,10,10,,,15,16,18,20,,,13,11,11,8,17,17,8,12,10,8,9,10,12,,,15,10,12,,14,8,13,23,21,12,,,,,11,12,,32,15,9,15,12,25,,19,12,12,12,12,10,9,12,11,10,11,11,31,12,13,,,9,10,18,15,,,22,,15,15,24,12,23,19,10,15,17,9,11,11
```
Ответ будет содержать CSV с предсказанными 111 STR в стандартном порядке FTDNA (палиндромы отделены друг от дуга и имеют суффиксы a, b, c, d).

### YFull Endpoint
Принимает на вход массив CSV, который можно получить в личном кабинете YFull на странице https://www.yfull.com/str/all/ путем нажатия кнопки "Загрузить .CSV" в правом верхнем углу. Ответ будет содержать CSV с предсказанными 111 STR в стандартном порядке FTDNA (палиндромы отделены друг от дуга и имеют суффиксы 1, 2, 3, 4).

### FTDNA Endpoint
Принимает на вход массив CSV (от 1 строки до сотен строк), который можно получить в личном кабинете FTDNA на странице https://www.familytreedna.com/my/y-dna-dys путем нажатия кнопки "CSV" в правом нижнем углу. Также, такой CSV можно собрать самостроятельно, скопировав из проекта FTDNA строку-заголовок с именами STR и интересующие строки с данными. Пример CSV-файла, использующего запятую в качестве разделителя:
```
YS393,DYS390,DYS19,DYS391,DYS385,DYS426,DYS388,DYS439,DYS389i,DYS392,DYS389ii,DYS458,DYS459,DYS455,DYS454,DYS447,DYS437,DYS448,DYS449,DYS464,DYS460,Y-GATA-H4,YCAII,DYS456,DYS607,DYS576,DYS570,CDY,DYS442,DYS438,DYS531,DYS578,DYF395S1,DYS590,DYS537,DYS641,DYS472,DYF406S1,DYS511,DYS425,DYS413,DYS557,DYS594,DYS436,DYS490,DYS534,DYS450,DYS444,DYS481,DYS520,DYS446,DYS617,DYS568,DYS487,DYS572,DYS640,DYS492,DYS565,DYS710,DYS485,DYS632,DYS495,DYS540,DYS714,DYS716,DYS717,DYS505,DYS556,DYS549,DYS589,DYS522,DYS494,DYS533,DYS636,DYS575,DYS638,DYS462,DYS452,DYS445,Y-GATA-A10,DYS463,DYS441,Y-GGAAT-1B07,DYS525,DYS712,DYS593,DYS650,DYS532,DYS715,DYS504,DYS513,DYS561,DYS552,DYS726,DYS635,DYS587,DYS643,DYS497,DYS510,DYS434,DYS461,DYS435
13,25,,10,12-13,12,13,10,12,11,28,15,9-10,11,11,24,14,20,33,13-15-16-16,10,10,19-23,15,16,18,20,33-36,13,11,11,8,17-17,8,12,10,8,9,10,,20-22,15,10,12,12,14,8,13,23,21,12,12,11,13,11,11,12,13,32,15,9,15,12,25,27,19,12,12,12,12,10,9,12,11,10,11,11,31,12,13,24,13,9,10,18,15,19,11,22,14,15,15,24,12,23,19,10,15,17,9,11,11
```
Ответ будет содержать CSV с предсказанными 111 STR в стандартном порядке FTDNA (палиндромы объединены друг с другом дефисами).

## Поддержка
Готов ответить на вопросы на форуме: https://forum.molgen.org/index.php/topic,14589.msg557146.html#msg557146
