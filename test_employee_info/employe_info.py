import os

def init_employee_info() -> str:
    prompt = """
    This function initializes an empty dictionary to store employee information.
    including: funding, adding, deleting and updating employee records.
    """
    return prompt.strip()

def get_employee_info()->dict:
    employe_infos = {}

    employe_file_data = open(r"D:\xyz\Learningcode\python\fundamentals\employee_info.txt", "r", encoding='utf-8')
    employe_file_line = employe_file_data.readline()
    while True:
        if not employe_file_line:
            break
        employe_info_line = employe_file_line.strip().split(",")
        employe_info = {}
        employe_info['id'] = employe_info_line[0]
        employe_info['name'] = employe_info_line[1]
        employe_info['age'] = employe_info_line[2]
        employe_info['telephone_number'] = employe_info_line[3]
        employe_info['dept'] = employe_info_line[4]
        employe_info['enroll_date'] = employe_info_line[5]
        employe_infos[employe_info['id']] = employe_info
        employe_file_line = employe_file_data.readline()
    employe_file_data.close()
    
    return employe_infos

def find_employe_info(choice:str) -> tuple:
    employe_infos = get_employee_info()
    find_employe_info = []
    for _, employe_info in employe_infos.items():
        if choice == '1':
            if int(employe_info['age']) > 22:
                find_employe_info.append([employe_info['name'], employe_info['age']])
        if choice == '2':
            if employe_info['dept'] == 'IT':
                find_employe_info.append(employe_info)
        if choice == '3':
            if '2013' in employe_info['enroll_date']:
                find_employe_info.append(employe_info)
    find_result = (len(find_employe_info), find_employe_info)
    return find_result

def add_employe_info() -> tuple:
    employe_infos = get_employee_info()
    print(f" id 从{len(employe_infos)+1}开始！\n")
    input_info = input("请输入要增加员工信息，输入格式<id,name,age,telephone_number,dept,enroll_date(eg:2025-12-25)>：")
    id, name, age, telephone_number, dept, enroll_date = input_info.strip().split(',')
    
    count = 0
    add_employe_ele = {}
    telephone_number_record = []
    for _, employe_info in employe_infos.items():
        telephone_number_record.append(employe_info['telephone_number'])
    import pdb;pdb.set_trace()
    if telephone_number in telephone_number_record:
        print("The phone number already exists. Please exit and enter it again.")
        exit()
    else:
        add_employe_ele['id'] = id
        add_employe_ele['name'] = name
        add_employe_ele['age'] = age
        add_employe_ele['telephone_number'] = telephone_number
        add_employe_ele['dept'] = dept
        add_employe_ele['enroll_date'] = enroll_date
        employe_infos[id] = add_employe_ele
        count += 1

        row = f"\n{id},{name},{age},{telephone_number},{dept},{enroll_date}"
        with open(r"D:\xyz\Learningcode\python\fundamentals\employee_info.txt", "a", encoding='utf-8') as f:
            f.write(row)
    find_result = (count, employe_infos)
    return find_result

def delete_employe_info() -> tuple:
    employe_infos = get_employee_info()
    id_record = []
    for key, _ in employe_infos.items():
        id_record.append(key)
    print("现有员工id: ", key)

    input_info = input("请输入要删除员工id(如果删除多个id，请以“，”分隔)：")
    if len(input_info) > 1:
        del_id = input_info.strip().split(',')
        for id in del_id:
            del employe_infos[id]
        count = len(del_id)
    else:
        id = input_info.strip()
        del employe_infos[id]
        count = 1
    find_result = (count, employe_infos)

    with open(r"D:\xyz\Learningcode\python\fundamentals\employee_info.txt", "w", encoding='utf-8') as f:
        for _, employe_info in employe_infos.items():
            line = f"{employe_info['id']},{employe_info['name']},{employe_info['age']},{employe_info['telephone_number']},{employe_info['dept']},{employe_info['enroll_date']} \n"
            f.write(line)
    return find_result

def update_employe_info() -> tuple:
    input_info = input("请输入要修改的信息，输入格式<修改对象，修改后的数值(eg: dept,IT)>：")
    item, change_data = input_info.strip().split(',')
    employe_infos = get_employee_info()
    count = 0
    for employe_id, employe_info in employe_infos.items():
        employe_info[item] = change_data
        count += 1
    find_result = (count, employe_infos)
    with open(r"D:\xyz\Learningcode\python\fundamentals\employee_info.txt", "w", encoding='utf-8') as f:
        for _, employe_info in employe_infos.items():
            line = f"{employe_info['id']},{employe_info['name']},{employe_info['age']},{employe_info['telephone_number']},{employe_info['dept']},{employe_info['enroll_date']} \n"
            import pdb;pdb.set_trace()
            f.write(line)
    return find_result



if __name__ == "__main__":

    employe_infos = get_employee_info()
    for key, value in employe_infos.items():
        print(f"{key}: {value}")
    
    choice='3'
    find_result = find_employe_info(choice)
    print(f"change_number: {find_result[0]}")
    for change_item in find_result[1]:
        print(f"{change_item}")

    add_result = add_employe_info()
    print(f"change_number: {add_result[0]}")
    for key, value in add_result[1].items():
        print(f"{key}: {value}")

    delete_result = delete_employe_info()
    print(f"change_number: {delete_result[0]}")
    for key, value in delete_result[1].items():
        print(f"{key}: {value}")


    update_result = update_employe_info()
    print(f"change_number: {update_result[0]}")
    for key, value in update_result[1].items():
        print(f"{key}: {value}")


