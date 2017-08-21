import os, sys
import webbrowser
from bs4 import BeautifulSoup
import email

def loadMail(mail_name):
    with open('../data/trec07p/data/' + mail_name, 'r') as f:
        mail = f.read().decode("ascii", errors = 'ignore')
    mail = email.message_from_string(mail)
    mail_content = ''
    # subject
    if 'Subject' in mail:
        mail_content += 'Subject: ' + mail['Subject'] + '\n'
    if mail.is_multipart():
        pl_list = getAllPayLoads(mail)
        for part in pl_list:
            conten_type = part.__getitem__('Content-Type')
            if conten_type and conten_type[:4].lower() == 'text':
                mail_content += part.get_payload() + '\n'
    else:
        conten_type = mail.__getitem__('Content-Type')
        if conten_type and conten_type[:4].lower() == 'text':
            mail_content += mail.get_payload() + '\n'
    return mail_content

def getAllPayLoads(mail):
    pl, pl_list = [mail], []
    while pl:
        p = pl.pop()
        if isinstance(p.get_payload(), list):
            pl.extend(p.get_payload())
        else:
            pl_list.append(p)
    return pl_list
    
if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        file_name = None
    else:
        file_name = args[1]

    html = loadMail(file_name)
    path = os.path.abspath('temp.html')
    url = 'file://' + path

    with open(path, 'w') as f:
        f.write(html)
    webbrowser.open(url)
