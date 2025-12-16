import requests
from typing import Optional
import datetime


BASE_URL = "https://www.funhpc.com"
LOGIN_ENDPOINT = "/api/user/passwordLogin"
LIST_ENDPOINT = "/api/instance/userList"
DELETE_ENDPOINT = "/api/instance/delete"

# 假设的用户凭证
PHONE = "13291037703"
PASSWORD = "zwy18117768535"



def get_auth_token(phone: str, password: str) -> Optional[str]:
    """尝试登录并返回JWT Token。"""
    login_url = BASE_URL + LOGIN_ENDPOINT
    login_payload = {
        "phone": phone,
        "pass": password
    }
    print(f"正在登录: {login_url} 使用手机号: {phone}")
    try:
        # 发送POST请求获取Token
        response = requests.post(
            login_url,
            json=login_payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()  # 对 4xx 或 5xx 状态码抛出异常

        data = response.json()
        if data.get("code") == 200 and "token" in data.get("data", {}):
            token = data["data"]["token"]
            print("✅ 登录成功，Token获取完毕。")
            return token
        else:
            print(f"❌ 登录响应错误: {data.get('msg', '未知错误')}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ 登录请求失败: {e}")
        return None


# --- 步骤 2: 获取实例 UUID ---
def get_instance_uuid(token: str) -> Optional[str]:
    """使用Token获取第一个实例的UUID。"""
    list_url = BASE_URL + LIST_ENDPOINT
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    print(f"正在查询实例列表: {list_url}")
    try:
        # 用户的 userList API 原型是一个 POST 请求，数据体为空 {}
        response = requests.post(list_url, headers=headers, json={})
        response.raise_for_status()

        data = response.json()
        if data.get("code") == 200 and data.get("data"):
            # 假设实例列表在 data['data'] 中，我们只取第一个
            instance_list = data["data"]
            if instance_list and len(instance_list) > 0:
                # 假设 uuid 字段名为 'uuid'
                instance_uuid = instance_list[0].get("uuid")
                if instance_uuid:
                    print(f"✅ 成功获取实例 UUID: {instance_uuid}")
                    return instance_uuid

            print("⚠️ 列表中没有找到可用的实例 UUID。")
            return None
        else:
            print(f"❌ 获取实例列表失败: {data.get('msg', '未知错误')}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ 获取实例列表请求失败: {e}")
        return None


# --- 步骤 3: 删除实例并保存数据 ---
def delete_instance(token: str, instance_uuid: str):
    """删除指定UUID的实例并设置数据保存。"""
    delete_url = BASE_URL + DELETE_ENDPOINT
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # 获取当前时间
    now = datetime.datetime.now()
    # 格式化时间为字符串：例如 '2025-12-05_225535'
    time_str = now.strftime("%Y-%m-%d_%H%M%S")

    # 您的要求是“删除并且保存数据”，对应 'data_save': true
    delete_payload = {
        "data_name": f"MSFMP_{time_str}",  # 根据您的原始请求，data_name 为空
        "data_save": True,  # 设置为 True 表示保存数据
        "uuid": instance_uuid
    }

    print(f"正在删除实例: {delete_url}, UUID: {instance_uuid}, 保存数据: True")
    try:
        response = requests.post(delete_url, headers=headers, json=delete_payload)
        response.raise_for_status()

        data = response.json()
        if data.get("code") == 200:
            print("🎉 实例删除请求成功，数据已标记为保存。")
        else:
            print(f"❌ 删除实例失败: {data.get('msg', '未知错误')}")

    except requests.exceptions.RequestException as e:
        print(f"❌ 删除实例请求失败: {e}")


# --- 主执行逻辑 ---
if __name__ == "__main__":

    # 1. 获取 Token
    jwt_token = get_auth_token(PHONE, PASSWORD)

    if jwt_token:
        # 2. 获取实例 UUID
        target_uuid = get_instance_uuid(jwt_token)

        if target_uuid:
            # 3. 删除实例
            delete_instance(jwt_token, target_uuid)
        else:
            print("🛑 无法继续，未找到目标实例 UUID。")
    else:
        print("🛑 无法继续，登录失败。")