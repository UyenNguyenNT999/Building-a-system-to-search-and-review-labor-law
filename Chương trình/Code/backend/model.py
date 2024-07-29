import pandas as pd
import networkx as nx
import re
from unidecode import unidecode

# class MetadataEmbedding:
#     def __init__(self, **kwargs) -> None:
#         self.model = get_model()
#         self.tokenizer = get_tokenizer()

#     def get_embedding_sentences(self, sentences: list) -> torch.Tensor:
#         sentences = list(map(clean_metadata, sentences))
#         try:
#             encoded_input = self.tokenizer(
#                 sentences,
#                 padding=True,
#                 truncation=True,
#                 max_length=300,
#                 return_tensors="pt",
#             ).to(device)
#         except RuntimeError:
#             self.tokenizer = get_tokenizer()
#             encoded_input = self.tokenizer(
#                 sentences,
#                 padding=True,
#                 truncation=True,
#                 max_length=128,
#                 return_tensors="pt",
#             ).to(device)

#         with torch.no_grad():
#             context = self.model(**encoded_input).last_hidden_state.mean(dim=1)
#         return context


class LawGraph:
    def __init__(self, data_file, sheet_name):
        self.df = pd.read_excel(data_file, sheet_name=sheet_name)
        self.G = nx.Graph()
        # self.meta_model = MetadataEmbedding()

    def build_graph(self):
        for _, row in self.df.iterrows():
            definition = str(row['dinh_nghia_clean']).lower()

            corresponding_rule = str(row['dieu_khoan_tuong_ung']).lower()
            content = str(row['noi_dung_chi_tiet']).lower()
            key_phrases = [phrase.strip() for phrase in row['key_pharse'].split(',')]

            self.G.add_node(definition, content=content, rule=corresponding_rule, key_phrases=key_phrases)

            # Xây dựng liên kết từ khóa đến định nghĩa
            for phrase in key_phrases:
                self.G.add_edge(phrase, definition)
    def jaccard_similarity(self,set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    def query_graph(self, question):
        matching_nodes = [node for node in self.G.nodes if node in question.lower()]
        question_words = set(question.lower().split())
        best_answer = None
        best_similarity = 0

        for node in matching_nodes:
            node_words = set(node.split())
            similarity = self.jaccard_similarity(node_words, question_words)
            if similarity > best_similarity:
                best_similarity = similarity
                best_answer = node

        if best_answer is not None:
            answer= self.G.nodes[best_answer].get('content', None)
            dieu_khoan = self.G.nodes[best_answer].get('rule', None)
            if dieu_khoan is None or answer is None:
                return "Không tìm thấy câu trả lời phù hợp."
            else:
                return f"Theo {dieu_khoan} thì \n {answer}"
        else:
            return None

            


# def clean_metadata(sentence: str):
#     if not sentence:
#         return "none"
#     sentence = convert_accented_vietnamese_text(sentence).lower()
#     sentence = re.sub('\s+', ' ', sentence).strip()
#     return sentence


# def get_tokenizer():
#     return RobertaTokenizerFast.from_pretrained(MODEL_PATH, max_len=512)


# def get_model():
#     model = torch.jit.load(
#         os.path.join(MODEL_PATH, "traced_bert_embedding_sentence.pt"),
#         map_location=device,
#     )

#     model.to(device)
#     model.eval()
#     return model

def preprocess_text(text):
    # Loại bỏ dấu và chuyển thành chữ thường
    text = unidecode(text.lower())
    # Loại bỏ các ký tự không phải là chữ cái và số
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def predict(user_question):
    data_file = 'data/data_final.xlsx'  # Thay đổi đường dẫn tới file dữ liệu

    # Định nghĩa từ điển ánh xạ từ khóa với sheet tương ứng
    keyword_to_sheet = {
        'bao hiem xa hoi': 'bao_hiem_xa_hoi',
        'bao hiem y te': 'bao_hiem_y_te',
        'bao hiem that nghiep': 'bao_hiem_that_nghiep'
    }

    # Đọc câu hỏi từ người dùng
    user_question = preprocess_text(user_question)

    # Tìm từ khóa trong câu hỏi và xác định sheet tương ứng
    matching_sheets = [sheet_name for keyword, sheet_name in keyword_to_sheet.items() if keyword in user_question]
    if not matching_sheets:
        return "Không tìm thấy từ khóa phù hợp trong câu hỏi."

    # Xây dựng knowledge graph và thực hiện truy vấn cho từng sheet tương ứng
    for sheet_name in matching_sheets:
        law_graph = LawGraph(data_file, sheet_name)
        law_graph.build_graph()

        predicted_answer = law_graph.query_graph(user_question)
        return predicted_answer
    
    return None

if __name__ == "__main__":
    # answer = predict("định nghĩa bảo hiểm thất nghiệp là gì?")
    # answer = predict("định nghĩa bảo hiểm xã hội là gì?")
    answer = predict("bảo hiểm xã hội là gì")
    print(answer)