import requests
import os
from typing import Optional, List, Dict
import datetime


# 增强版：多方式获取主机名核心标识（解决HOSTNAME环境变量获取失败问题）
def get_target_uuid_from_hostname() -> Optional[str]:
    try:
        with open("/proc/sys/kernel/hostname", "r") as f:
            full_hostname = f.read().strip()  # 读取完整主机名
            # 按'-'分割后，取前两段拼接（核心UUID）
            hostname_parts = full_hostname.split('-')
            if len(hostname_parts) >= 2:
                target_uuid = f"{hostname_parts[0]}-{hostname_parts[1]}"
                print(f"✅ 从/proc文件提取目标UUID：{target_uuid}")  # 输出：xaosepnwmbobnbvg-snow
                return target_uuid
            else:
                print(f"❌ 主机名格式异常，无法提取前两段：{full_hostname}")
                return None
    except (FileNotFoundError, PermissionError, IOError) as e:
        print(f"⚠️ 从/proc文件提取执行失败：{e}")
        return None


# 全局目标UUID
TARGET_UUID = get_target_uuid_from_hostname()



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


# --- 步骤 2: 获取所有实例 ---
def get_all_instances(token: str) -> Optional[List[Dict]]:
    """获取所有实例的完整列表（返回原始数据），方便后续筛选"""
    list_url = BASE_URL + LIST_ENDPOINT
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    print(f"\n正在查询所有实例列表: {list_url}")
    try:
        response = requests.post(list_url, headers=headers, json={})
        response.raise_for_status()

        data = response.json()
        if data.get("code") == 200 and isinstance(data.get("data"), list):
            instances = data["data"]
            print(f"✅ 共获取到 {len(instances)} 个实例")
            return instances
        else:
            print(f"❌ 获取实例列表失败: {data.get('msg', '未知错误')}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ 获取实例列表请求失败: {e}")
        return None

# --- 步骤 3: 找到匹配的实例uuid ---
def filter_my_instance(instances: List[Dict]) -> Optional[str]:
    """遍历所有实例，匹配指定UUID并返回对应的实例ID"""
    # 校验实例列表是否为空
    if not instances:
        print("⚠️ 实例列表为空，无法筛选")
        return None

    print(f"\n🔍 开始遍历实例，匹配目标UUID：{TARGET_UUID}")
    print(f"📊 待遍历实例总数：{len(instances)}")

    # 遍历所有实例，精准匹配UUID
    for idx, inst in enumerate(instances, 1):
        # 获取当前实例的UUID（兼容大小写/空值）
        current_uuid = inst.get("uuid", "").strip()
        # 获取当前实例的ID（需确认字段名，常见为"id"/"instance_id"，可根据实际调整）
        instance_id = inst.get("id")  # 核心：实例ID字段名，需按实际返回值调整

        print(f"\n实例 {idx} 检查：")
        print(f"  当前UUID: {current_uuid}")
        print(f"  当前实例ID: {instance_id}")

        # 精准匹配目标UUID
        if current_uuid == TARGET_UUID:
            if instance_id:
                print(f"✅ 找到匹配UUID的实例！")
                print(f"  匹配UUID: {TARGET_UUID}")
                print(f"  对应实例ID: {instance_id}")
                return TARGET_UUID
            else:
                print(f"❌ 匹配到目标UUID，但该实例无「id」字段！")
                return None

    # 遍历结束未找到匹配的UUID
    print(f"\n❌ 遍历所有实例后，未找到UUID等于「{TARGET_UUID}」的实例")
    return None

# --- 步骤 4: 删除实例并保存数据 ---
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
        # 2. 获取所有实例列表
        all_instances = get_all_instances(jwt_token)

        if all_instances:
            # 3. 筛选目标实例（自己的实例）
            target_uuid = filter_my_instance(all_instances)
            if target_uuid:
                # 4. 直接删除实例（移除y/n确认）
                print("\n📌 开始执行实例删除操作...")
                delete_instance(jwt_token, target_uuid)
                print("\n📌 实例删除完毕...")
            else:
                print("🛑 无法继续，未找到目标实例。")
        else:
            print("🛑 无法继续，获取实例列表失败。")
    else:
        print("🛑 无法继续，登录失败。")