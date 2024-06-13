from ftplib import FTP

import time
import tarfile


def ftpconnect(host, username, password):
    ftp = FTP()
    ftp.set_debuglevel(2)
    ftp.connect(host, 21)
    ftp.login(username, password)

    return ftp


def downloadfile(ftp, remotepath, localpath):
    bufsize = 1024
    fp = open(localpath, 'wb')
    ftp.retrbinary('RETR  ' + remotepath, fp.write, bufsize)
    # 接受服务器上文件并写入文本
    ftp.set_debuglevel(0)  # 关闭调试
    fp.close()  # 关闭文件


def uploadfile(ftp, remotepath, localpath):
    bufsize = 1024
    fp = open(localpath, 'rb')
    ftp.storbinary('STOR ' + remotepath, fp, bufsize)  # 上传文件
    #ftp.set_debuglevel(0)
    fp.close()
