import pandas as pd
import pandas as pd
from io import BytesIO

class Model :
    def __init__(self,model,columns,target,scaler):
        self.model = model
        self.columns = columns
        self.target = target
        self.scaler = scaler


def get_type(path:str)->str:
    w = path.split('.')
    return w[-1]


def readfile(path):
    type = get_type(path.name)
    if type == 'csv':
        return pd.read_csv(path)
    if type in ["xls" , "xlsx" , "xlsm"]:
        return pd.read_excel(path)
    if type == 'json':
        return pd.read_json(path)
    if type == 'xml':
        return pd.read_xml(path)
    

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.close()
    processed_data = output.getvalue()
    return processed_data

def exportfile(data:pd.DataFrame,type):
    if type == 'csv':
        return data.to_csv(index=False)
    if type in ["xlsx" , "xlsm",'xls']:
        return to_excel(data)
    if type == 'json':
        return data.to_json(index=False)
    if type == 'xml':
        return data.to_xml(index=False)