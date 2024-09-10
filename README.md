# tamrin-9.4.2024-Tree-Method-
(Tree Method) tamrin 9.4.2024 github
!pip install pyspark
from pyspark.sql import SparkSession as ss
spark=ss.builder.appName('3metodtemrin').getOrCreate()
from google.colab import drive
drive.mount('/content/drive')
df=spark.read.csv('/content/drive/MyDrive/College (1).csv',inferSchema=True,header=True)
df.show()
df.describe()
df.printSchema()
from pyspark.ml.linalg import Vectors as vc
from pyspark.ml.feature import VectorAssembler as va
df.columns
assembler = va(
  inputCols=['Apps',
             'Accept',
             'Enroll',
             'Top10perc',
             'Top25perc',
             'F_Undergrad',
             'P_Undergrad',
             'Outstate',
             'Room_Board',
             'Books',
             'Personal',
             'PhD',
             'Terminal',
             'S_F_Ratio',
             'perc_alumni',
             'Expend',
             'Grad_Rate'],
              outputCol="features")
output=assembler.transform(df)
output.show()
from pyspark.ml.feature import StringIndexer as siuu
ndxr = siuu(inputCol="Private",outputCol="PrivateIndex")
outputfixed = ndxr.fit(output).transform(output)
outputfixed.head(1)
finaldata = outputfixed.select("features","PrivateIndex")
finaldata.show()
traindata,testdata = finaldata.randomSplit([0.7,0.3])
