"""這台電腦在區網裡的 IPv4 位址：設定頁和啟動畫面顯示「手機、平板要連哪個網址」用。

用 Windows 的 GetAdaptersAddresses 列出網卡，排除虛擬網卡（VMware、VirtualBox、Hyper-V、WSL、VPN、
Tailscale、藍牙、Wi-Fi Direct……），只留啟用中、有預設閘道的有線或 Wi-Fi 網卡。
一個都找不到時退回列出全部非本機位址，免得什麼都不顯示。
"""
import ctypes
import os
import socket

IF_TYPE_ETHERNET = 6
IF_TYPE_WIFI = 71
OPER_STATUS_UP = 1
VIRTUAL_WORDS = ("virtual", "vmware", "virtualbox", "hyper-v", "vethernet", "wsl", "docker", "tap-windows",
                 "tap adapter", "tailscale", "wireguard", "wintun", "openvpn", "zerotier", "hamachi", "radmin",
                 "npcap", "loopback", "bluetooth", "藍牙", "wi-fi direct", "vpn")


class _SockAddr(ctypes.Structure):
    _fields_ = [("sa_family", ctypes.c_ushort), ("sa_data", ctypes.c_ubyte * 14)]


class _SocketAddress(ctypes.Structure):
    _fields_ = [("lpSockaddr", ctypes.POINTER(_SockAddr)), ("iSockaddrLength", ctypes.c_int)]


class _Unicast(ctypes.Structure):
    pass


_Unicast._fields_ = [("Length", ctypes.c_ulong), ("Flags", ctypes.c_ulong), ("Next", ctypes.POINTER(_Unicast)),
                     ("Address", _SocketAddress)]


class _Gateway(ctypes.Structure):
    pass


_Gateway._fields_ = [("Length", ctypes.c_ulong), ("Reserved", ctypes.c_ulong), ("Next", ctypes.POINTER(_Gateway)),
                     ("Address", _SocketAddress)]


class _Adapter(ctypes.Structure):
    pass


# IP_ADAPTER_ADDRESSES_LH（64 位元），只用到 FirstGatewayAddress 為止
_Adapter._fields_ = [
    ("Length", ctypes.c_ulong), ("IfIndex", ctypes.c_ulong), ("Next", ctypes.POINTER(_Adapter)),
    ("AdapterName", ctypes.c_char_p), ("FirstUnicastAddress", ctypes.POINTER(_Unicast)),
    ("FirstAnycastAddress", ctypes.c_void_p), ("FirstMulticastAddress", ctypes.c_void_p),
    ("FirstDnsServerAddress", ctypes.c_void_p), ("DnsSuffix", ctypes.c_wchar_p), ("Description", ctypes.c_wchar_p),
    ("FriendlyName", ctypes.c_wchar_p), ("PhysicalAddress", ctypes.c_ubyte * 8), ("PhysicalAddressLength", ctypes.c_ulong),
    ("Flags", ctypes.c_ulong), ("Mtu", ctypes.c_ulong), ("IfType", ctypes.c_ulong), ("OperStatus", ctypes.c_int),
    ("Ipv6IfIndex", ctypes.c_ulong), ("ZoneIndices", ctypes.c_ulong * 16), ("FirstPrefix", ctypes.c_void_p),
    ("TransmitLinkSpeed", ctypes.c_ulonglong), ("ReceiveLinkSpeed", ctypes.c_ulonglong),
    ("FirstWinsServerAddress", ctypes.c_void_p), ("FirstGatewayAddress", ctypes.POINTER(_Gateway)),
]

_GAA_FLAGS = 0x2 | 0x4 | 0x8 | 0x80   # 略過 anycast、multicast、DNS，包含閘道


def _ipv4(sock_address: _SocketAddress) -> str | None:
    if not sock_address.lpSockaddr:
        return None
    sa = sock_address.lpSockaddr.contents
    if sa.sa_family != socket.AF_INET:
        return None
    return socket.inet_ntoa(bytes(sa.sa_data[2:6]))


def adapters() -> list[dict]:
    """每張網卡：name、description、type（6 有線、71 Wi-Fi）、up、ipv4、gateways。"""
    if os.name != "nt":
        return []
    size = ctypes.c_ulong(16000)
    iphlpapi = ctypes.windll.iphlpapi
    for _ in range(4):
        buf = ctypes.create_string_buffer(size.value)
        rc = iphlpapi.GetAdaptersAddresses(socket.AF_INET, _GAA_FLAGS, None, buf, ctypes.byref(size))
        if rc != 111:   # ERROR_BUFFER_OVERFLOW：size 已經改成需要的大小，再試一次
            break
    if rc != 0:
        return []
    out = []
    p = ctypes.cast(buf, ctypes.POINTER(_Adapter))
    while p:
        a = p.contents
        ips, gateways = [], []
        u = a.FirstUnicastAddress
        while u:
            ip = _ipv4(u.contents.Address)
            if ip:
                ips.append(ip)
            u = u.contents.Next
        g = a.FirstGatewayAddress
        while g:
            ip = _ipv4(g.contents.Address)
            if ip and ip != "0.0.0.0":
                gateways.append(ip)
            g = g.contents.Next
        out.append({"name": a.FriendlyName or "", "description": a.Description or "", "type": a.IfType,
                    "up": a.OperStatus == OPER_STATUS_UP, "ipv4": ips, "gateways": gateways})
        p = a.Next
    return out


def is_virtual(adapter: dict) -> bool:
    text = f"{adapter.get('name', '')} {adapter.get('description', '')}".lower()
    return any(word in text for word in VIRTUAL_WORDS)


def _usable(ip: str) -> bool:
    return not ip.startswith(("127.", "169.254.", "0."))


def _primary_ip() -> str | None:
    """對外連線時實際會用的那張網卡（不會真的送出封包）。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.168.0.1", 9))
            return s.getsockname()[0]
    except OSError:
        return None


def pick_lan_addresses(items: list[dict], primary: str | None = None) -> list[str]:
    """從網卡清單挑要顯示的位址（拆出來方便測試）。"""
    def collect(pred):
        seen = []
        for a in items:
            if pred(a):
                for ip in a["ipv4"]:
                    if _usable(ip) and ip not in seen:
                        seen.append(ip)
        return seen

    ips = collect(lambda a: a["up"] and a["type"] in (IF_TYPE_ETHERNET, IF_TYPE_WIFI) and a["gateways"]
                  and not is_virtual(a))
    if not ips:   # 做不好就先列出全部：啟用中的非虛擬網卡，再不行就全部
        ips = collect(lambda a: a["up"] and not is_virtual(a)) or collect(lambda a: a["up"])
    return sorted(ips, key=lambda ip: (ip != primary, ips.index(ip)))


def lan_addresses() -> list[str]:
    try:
        ips = pick_lan_addresses(adapters(), _primary_ip())
    except (OSError, ValueError, AttributeError):
        ips = []
    if ips:
        return ips
    # 讀不到網卡清單（非 Windows 或 API 失敗）：用電腦名稱查位址
    found = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if _usable(ip) and ip not in found:
                found.append(ip)
    except OSError:
        pass
    return found
