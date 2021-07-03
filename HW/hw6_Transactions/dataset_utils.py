import numpy as np
import pandas as pd
import pickle
import gc
import datetime

def feature_generation(data):
    
    data['Errors?'] = data['Errors?'].fillna('').str.strip(',').str.split(',')
    exploded_errors = data['Errors?'].explode()
    error_cause = pd.get_dummies(exploded_errors).groupby(exploded_errors.index).sum()

    del exploded_errors
    gc.collect()
    
    data = data.drop(['Errors?'], axis=1).merge(error_cause, left_index=True,
                                                 right_index=True, suffixes=(None, None))

    del error_cause
    gc.collect()
    
    data['Amount'] = data.Amount.str.replace("$", "").astype(float)
    gc.collect()
    
    time = pd.DataFrame(data.Time.str.split(':').to_list()).astype('int64').values
    data[['Hour', 'Minute']] = time
    
    del time
    
    data.drop(['Time', 'Merchant Name', 'Merchant City',
                     'Zip', 'Is Fraud?'], axis=1, inplace=True)
    
    gc.collect()
    
    data['Weekday'] = data[['Year', 'Month', 'Day']].apply(lambda x: datetime.datetime(*x).weekday(), axis=1)
    data['Merchant State'] = data['Merchant State'].fillna("ONLINE")
    data['Weekend'] = data.Weekday.ge(5)
    data['Daytime'] = pd.cut(data.Hour, 4, labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    gc.collect()
    
    data['Weekday_cos'] = np.cos(2 * np.pi / 7.) * data.Weekday
    data['Weekday_sin'] = np.sin(2 * np.pi / 7.) * data.Weekday
    data['Weekday_tan'] = np.tan(2 * np.pi / 7.) * data.Weekday

    data['Hour_cos'] = np.cos(2 * np.pi / 24.) * data.Hour
    data['Hour_sin'] = np.sin(2 * np.pi / 24.) * data.Hour
    data['Hour_tan'] = np.tan(2 * np.pi / 24.) * data.Hour

    data['Month_cos'] = np.cos(2 * np.pi / 12.) * data.Month
    data['Month_sin'] = np.sin(2 * np.pi / 12.) * data.Month
    data['Month_tan'] = np.tan(2 * np.pi / 12.) * data.Month
    
    gc.collect()
    
    return data

    
    
def preprocessing(link):

    embedding_columns = ['Card', 'Use Chip', 'Merchant State', 'MCC', 'Bad CVV',
       'Bad Card Number', 'Bad Expiration', 'Bad PIN', 'Bad Zipcode',
       'Insufficient Balance', 'Technical Glitch', 'Weekday', 'Weekend', 'Daytime']
    
    nrof_emb_categories = {}
    unique_categories = {}
    
    print("start")

    with open(link, 'rb') as f:
        dictionary = pickle.load(f)
        
        train_data = dictionary['train']
        valid_data = dictionary['valid']
        test_data = dictionary['test']
        
        del dictionary
        gc.collect()
        
        print('data collected')
        
    train_data = feature_generation(train_data)
    valid_data = feature_generation(valid_data)
    test_data = feature_generation(test_data)
    gc.collect()

    print('features generated')

    for cat in embedding_columns:
        nrof_unique = np.unique(train_data[cat].unique().astype(np.str).tolist() +
                                valid_data[cat].unique().astype(np.str).tolist() +
                                test_data[cat].unique().astype(np.str).tolist())
        
        gc.collect()
        
        unique_categories[cat] = nrof_unique
        nrof_emb_categories[cat] = len(nrof_unique)
        
        train_data[cat + '_cat'] = [np.where(nrof_unique == val)[0][0] for i, val in enumerate(train_data[cat].values.astype(np.str))]
        valid_data[cat + '_cat'] = [np.where(nrof_unique == val)[0][0] for i, val in enumerate(valid_data[cat].values.astype(np.str))]
        test_data[cat + '_cat'] = [np.where(nrof_unique == val)[0][0] for i, val in enumerate(test_data[cat].values.astype(np.str))]
        
        train_data.drop([cat], axis=1, inplace=True)
        valid_data.drop([cat], axis=1, inplace=True)
        test_data.drop([cat], axis=1, inplace=True)

        gc.collect()

    with open('train.pickle', 'wb') as f:
        pickle.dump([train_data, nrof_emb_categories, unique_categories], f)
        
    del train_data
    gc.collect()

    print('train generated')
    
    with open('valid.pickle', 'wb') as f:
        pickle.dump([valid_data, nrof_emb_categories, unique_categories], f)
        
    del valid_data
    gc.collect()

    print('valid generated')
    
    with open('test.pickle', 'wb') as f:
        pickle.dump([test_data, nrof_emb_categories, unique_categories], f)
    
    del test_data
    gc.collect()

    print('test generated')