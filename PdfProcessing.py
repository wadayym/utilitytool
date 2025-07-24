import PyPDF2
from pdfminer.high_level import extract_text
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import re

# PDFを回転する
def pdf_roll(p_file, p_angle, r_file):
    file = PyPDF2.PdfReader(open(p_file, 'rb'))
    file_output = PyPDF2.PdfWriter()
    for page_num in range(file.numPages):
        page = file.getPage(page_num)
        page.rotateClockwise(p_angle)
        file_output.addPage(page)
    with open(r_file, 'wb') as f:
        file_output.write(f)

# PDFをテキストにする
def pdf2text(p_file, t_file):
    f = open(t_file, "w", encoding='utf-8')    
    print(extract_text(p_file), file=f)
    removeLineFeed(t_file)

# 縦書きのPDFをテキストにする
def pdfvertical2text(p_file, t_file):
    pdfname = p_file
    ret = gettext(pdfname)
    file = open(t_file, 'w', encoding='utf-8')
    file.write(ret)
    file.close()
    removeLineFeed(t_file)


def gettext(pdfname):
    # PDFファイル名が未指定の場合は、空文字列を返して終了
    if (pdfname == ''):
        return ''
    else:
        # 処理するPDFファイルを開く/開けなければ
        try:
            fp = open(pdfname, 'rb')
        except:
            return ''

    # リソースマネージャインスタンス
    rsrcmgr = PDFResourceManager()
    # 出力先インスタンス
    outfp = StringIO()
    # パラメータインスタンス
    laparams = LAParams()
    # 縦書き文字を横並びで出力する
    laparams.detect_vertical = True
    # デバイスの初期化
    device = TextConverter(rsrcmgr, outfp, codec='utf-8', laparams=laparams)
    # テキスト抽出インタプリタインスタンス
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # 対象ページを読み、テキスト抽出する。（maxpages：0は全ページ）
    for page in PDFPage.get_pages(fp, pagenos=None, maxpages=0, password=None,\
            caching=True, check_extractable=True):
        interpreter.process_page(page)
    #取得したテキストをすべて読みだす
    ret = outfp.getvalue()
    # 後始末をしておく    
    fp.close()
    device.close()
    outfp.close()

    return ret

#改行だけの行を削除する
def removeLineFeed(t_file):
    file = open(t_file, 'r' ,encoding='utf-8')
    lines = file.readlines()
    file.close()

    file = open(t_file, 'w', encoding='utf-8')
    text = []
    for line in lines:
        line = re.sub(r'^\n$', '', line)
        text.append(line)
    file.writelines(text)
    file.close()