import torch
from model import RnnEncoder, Classifier
from transformers import AutoTokenizer 


tokenizer = AutoTokenizer.from_pretrained("alibayram/tr_tokenizer")
one_step_encoder = RnnEncoder()
classifier = Classifier()

# A paragraph -> 128 tensor

text = '''
14 Ağustos 2001'de kurulan Adalet ve Kalkınma Partisi'nin kurucuları arasında yer aldı ve partinin genel başkanlığına seçildi. Parti, girdiği ilk seçimler olan 2002 genel seçimlerinde %34,43'lük oy oranı ile Abdullah Gül'ün başbakanlığında 58. hükûmeti kurarken, siyasi yasağı süren Erdoğan seçimlere girememişti. Siyasi yasağının kaldırılması için Türkiye Büyük Millet Meclisine sunulan yasa değişikliği talebinin uygulamaya girmesiyle siyasi yasağı kalktı. 9 Mart 2003'te gerçekleştirilen ara seçimlerde Siirt milletvekili olarak meclise girdi. Başbakan Gül'ün istifasını sunmasıyla, 14 Mart 2003'te başbakanlık görevine geldi. Genel başkanlığını yürüttüğü Adalet ve Kalkınma Partisi, 2007 genel seçimlerinde oyların %46,58, 2011 genel seçimlerinde ise oyların %49,83'ünü alarak Erdoğan'ın Başbakanlığı'nda sırasıyla 60. ve 61. hükûmetleri kurdu. Sonraki süreçte yapılan genel seçimlerde partisi 2018'de %42,56,[7] 2023'te ise %35,62[8][9] oy alarak en yüksek milletvekili sayısına ulaştı. Parti ayrıca, oyların %41,67'sini aldığı 2004 yerel seçimleri, oyların %38,39'unu aldığı 2009 yerel seçimleri, oyların %43,40'ını aldığı 2014 yerel seçimlerinde ve oyların %42,56'sını aldığı 2019 yerel seçimlerinde[10] de en çok oy toplamayı başaran parti konumundaydı. 2007 anayasa değişikliği referandumu sonrasında anayasada yapılan değişiklikle birlikte cumhurbaşkanının ilk defa doğrudan halk oyuyla seçilmesinin önü açılırken, adaylığını koyduğu 10 Ağustos 2014'te yapılan seçimlerde aldığı %51,79'luk oy oranıyla cumhurbaşkanı seçildi ve başbakanlık ile partisindeki görevinden ayrılarak Cumhurbaşkanlığı görevine 28 Ağustos 2014'te başladı. 2018 cumhurbaşkanlığı seçiminde oyların %52,59'unu aldı ve ikinci kez cumhurbaşkanı seçildi.[11] 2023 Cumhurbaşkanlığı seçiminde de oyların %52,18'ini alarak üçüncü kez cumhurbaşkanı seçildi.[12] 3 Haziran 2023 tarihinde Cumhurbaşkanlığı Külliyesi'nde yapılan göreve başlama töreni sonrasında görevine başladı.[13]
'''

tokenized_text = torch.tensor(tokenizer(text)['input_ids'])

h = torch.Tensor(128, 1)
for idx, token_id in enumerate(tokenized_text):
    print(tokenizer.decode(token_id))
    h = one_step_encoder(token_id, h)

logits = classifier(h).view(-1)
softmax = torch.nn.Softmax(dim=0)
y = softmax(logits)
print(y)
