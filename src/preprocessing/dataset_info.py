def get_dataset_info():
    subject_list = ['01',
                    '03',
                    '04',
                    '06',
                    '07',
                    '11',
                    '12',
                    '13',
                    '14',
                    '15',
                    '17',
                    '19',
                    '20',
                    '22',
                    '24',
                    '25',
                    '26',
                    '27',
                    '28',
                    '29',
                    '30',
                    '31',
                    '32',
                    '33',
                    '34',
                    '35',
                    '36',
                    '39',
                    '41',
                    '42',
                    '43',
                    '44',
                    '46',
                    '47',
                    '48',]
    sub_to_StimList = {}
    for i in range(27):
        sub_to_StimList[subject_list[i]] = subject_list[i]
    sub_to_StimList['39'] = '08'
    sub_to_StimList['41'] = '16'
    sub_to_StimList['42'] = '18'
    sub_to_StimList['43'] = '21'
    sub_to_StimList['44'] = '23'
    sub_to_StimList['46'] = '05'
    sub_to_StimList['47'] = '02'
    sub_to_StimList['48'] = '10'
    return subject_list, sub_to_StimList

if __name__ == '__main__':
    get_dataset_info()
