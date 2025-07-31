
import sys, os
import json

MASHUP_DATA_PATH = 'data/mashupData.json'
API_DATA_PATH = 'apiData.json'
with open(MASHUP_DATA_PATH, 'r') as fd:
    mashupRawData = fd.read()
    mashupData = json.loads(mashupRawData)
with open(API_DATA_PATH, 'r') as fd:
    apiRawData = fd.read()
    apiData = json.loads(apiRawData)



n_minimum=3
n_maximum=10000

#样本中 保留调用api数>=n 的mashup
data_list,api_dict=data_pro(MASHUP_DATA_PATH,API_DATA_PATH,n_minimum,n_maximum,mashup_need_feat=['Name','Tags','Related APIs','Description','Type'],api_need_feat=['Name','Primary Category','Secondary Categories'])



# 全局变量
# FlagFeatConvert：用于 是否进行 名称对照操作
FlagFeatConvert = False

# 更改特征名称
'''定义特征名称对照表
['Name', 'Description', 'FollowerNum', 'API Endpoint', 'API Portal / Home Page', 'Primary Category', 'Secondary Categories', 'Version status', 'Terms Of Service URL', 'Is the API Design/Description Non-Proprietary ?', 'Scope', 'Device Specific', 'Docs Home Page URL', 'Architectural Style', 'Supported Request Formats', 'Supported Response Formats', 'Is This an Unofficial API?', 'Is This a Hypermedia API?', 'Restricted Access ( Requires Provider Approval )', 'TagList', 'Id', 'SSL Support', 'API Forum / Message Boards', 'Support Email Address', 'Developer Support URL', 'API Provider', 'Twitter URL', 'Authentication Model', 'Type', 'Version', 'Description File URL', 'Description File Type', 'How is this API different ?', 'Is the API related to anyother API ?', 'Interactive Console URL', 'Developer Home Page', 'Type of License if Non Proprietary', 'Streaming Technology', 'Streaming Directions', 'Direction Selection']
['Name', 'Submitted Date', 'Related APIs', 'Tags', 'Url', 'Company', 'Type', 'FollowerNum', 'Description', 'APIs', 'TagList', 'Id']
'''
featTable = {} 
featTable['mashup'] = {
    "Id":"MashupId",
    "Name":"MashupName",
    "TagList":"MashupTags",
    "Category":"MashupCategory",

    "Description":"MashupDescription",
    "Submitted Date":"MashupSubDate",
    "Url":"MashupUrl",
    "Company":"MashupCompany",
    "Type":"MashupType",
    "FollowerNum":"MashupFollowerNum",
    "APIs":"MashupRelatedAPIs"
}
featTable['api'] = {
    "Id":"ApiId",
    "Name":"ApiName",
    "TagList":"ApiTags",
    "Primary Category":"ApiCategory",
    "Description":"ApiDescription",
    "FollowerNum":"ApiFollowerNum",
    "API Endpoint":"ApiEndpoint",
    "API Portal / Home Page":"ApiHome",
    "Version status":"ApiVersionStatus",
    "Terms Of Service URL":"ApiTermsOfServiceUrl",
    "Is the API Design/Description Non-Proprietary ?":"ApiNonProprietary",
    "Scope":"ApiScope",
    "Device Specific":"ApiDeviceSpecific",
    "Docs Home Page URL":"ApiDocsHome",
    "Architectural Style":"ApiArchitecture",
    "Supported Request Formats":"ApiRequestFormats",
    "Supported Response Formats":"ApiResponseFormats",
    "Is This an Unofficial API?":"ApiUnofficial",
    "Is This a Hypermedia API?":"ApiHypermedia",
    "Restricted Access ( Requires Provider Approval )":"ApiRestrictedAccess",
    "SSL Support":"ApiSSLSupport",
    "API Forum / Message Boards":"ApiForum",
    "Support Email Address":"ApiSupportEmail",
    "Developer Support URL":"ApiDeveloperSupportURL",
    "API Provider":"ApiProvider",
    "Twitter URL":"ApiTwitterURL",
    "Authentication Model":"ApiAuthModel",
    "Type":"ApiType",
    "Version":"ApiVersion",
    "Description File URL":"ApiDescriptionFileURL",
    "Description File Type":"ApiDescriptionFileType",
    "How is this API different ?":"ApiHowDifferent",
    "Is the API related to anyother API ?":"ApiRelated2AnyotherAPI",
    "Interactive Console URL":"ApiInteractiveConsoleUrl",
    "Developer Home Page":"ApiDeveloperHome",
    "Type of License":"ApiTypeOfLicense",
    "Streaming Technology":"ApiStreamingTechnology",
    "Streaming Directions":"ApiStreamingDirections",
    "Direction Selection":"ApiDirectionSelection"
}



# 删除没有des描述信息的mashup：
new_data_list=[]
for i in range(len(data_list)):
    if len(data_list[i]['Description'])!=0:
        new_data_list.append(data_list[i])
        
data_list=new_data_list



# 进行名称映射
new_data_list=[]
# 将datalist中的mashup进行ID编码, 写到csv文件中
for i in range(len(data_list)):
    temp_dict={}
    temp_dict['ID']=i
    for key,value in data_list[i].items():
        temp_dict[featTable['mashup'][key]]=value
    new_data_list.append(temp_dict)
    
data_list=new_data_list



import csv
keys=data_list[0].keys()
filename = 'data/final_data/mashup_data.csv'

with open(filename, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=keys)
    writer.writeheader()  # 写入CSV文件的标题行
    writer.writerows(data_list)  # 写入字典列表的数据行



# 检验mashup数据
data_name_list=[]
for i in data_list:
    if i["MashupName"] in data_name_list:
        data_name_list.append(i["MashupName"])
        print("wrong1")
    k=list(set(i["MashupRelatedAPIs"]))
    if len(k)!=len(i["MashupRelatedAPIs"]):
        print("wrong2")



# 生成apidata：  单独处理taglist
api_list=[]
api_need_feats=["Name","Description","SSL Support","API Provider","Authentication Model","Is the API Design/Description Non-Proprietary ?"]
api_name_list=[]

index=0
for md in data_list:
    for ad_name in md['MashupRelatedAPIs']:
        if ad_name not in api_name_list:
            api_name_list.append(ad_name)
            # 遍历所有的apidata查找该api
            for ad in apiData:
                if ad["Name"]==ad_name:
                    temp_dict={}
                    temp_dict['ID']=index
                    index+=1
                    for need_feat in api_need_feats:
                        if need_feat in ad:
                            temp_dict[featTable['api'][need_feat]]=ad[need_feat]
                        else:
                            temp_dict[featTable['api'][need_feat]]="null"
                    break
            api_list.append(temp_dict)





# 检验api数据
api_name_list=[]
for i in api_list:
    if i["ApiName"] in data_name_list:
        data_name_list.append(i["ApiName"])
        print("wrong")



# 处理 taglit：
for i in range(len(api_list)):
    api_list[i]["ApiTags"]=api_dict[api_list[i]["ApiName"]]




# 将API信息写入csv：
keys=api_list[0].keys()
filename = 'data/final_data/api_data.csv'

with open(filename, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=keys)
    writer.writeheader()  # 写入CSV文件的标题行
    writer.writerows(api_list)  # 写入字典列表的数据行



# 调用关系 mashupid,relation,apiid:
rel_list=[]
for md in data_list:
    for ad_name in md['MashupRelatedAPIs']:
        for iter_ad in api_list:
            if ad_name==iter_ad["ApiName"]:
                temp_dict={}
                temp_dict["Mashup_ID"]=md['ID']
                temp_dict["Relation"]='Invoke'
                temp_dict["Api_ID"]=iter_ad['ID']
                break
            
        rel_list.append(temp_dict)    


# 将调用关系写入csv：
keys=rel_list[0].keys()
filename = 'data/final_data/Invoke_edge.csv'

with open(filename, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=keys)
    writer.writeheader()  # 写入CSV文件的标题行
    writer.writerows(rel_list)  # 写入字典列表的数据行