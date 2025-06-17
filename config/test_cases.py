from pathlib import Path
from benchmark.utils import create_qwen_prompt

MMLU_TEST_DIR = Path("./mmlu_test")

# Các đoạn văn bản để kiểm tra Perplexity
PERPLEXITY_CATEGORIES = {
    # 1. General Knowledge & Factual Statements
    # Đo lường khả năng của model với các câu khẳng định, kiến thức phổ thông.
    # PPL ở đây nên thấp.
    "general_knowledge": [
        "The capital of France is Paris, a major European city and a global center for art, fashion, gastronomy, and culture.",
        "Water is a chemical compound with the formula H2O, consisting of one oxygen atom and two hydrogen atoms.",
        "The Earth revolves around the Sun in an elliptical orbit, completing one full revolution in approximately 365.25 days.",
        "William Shakespeare was an English playwright, poet, and actor, widely regarded as the greatest writer in the English language.",
        "The Internet is a global network of interconnected computers that use the Internet protocol suite (TCP/IP) to link devices worldwide.",
        "Mount Everest, part of the Himalayas, is the Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range.",
        "The human heart is a muscular organ responsible for pumping blood through the blood vessels by repeated, rhythmic contractions.",
        "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy, through a cellular process."
    ],

    # 2. Scientific & Technical Language
    # Đo lường khả năng xử lý thuật ngữ chuyên ngành. Model được huấn luyện tốt về khoa học sẽ có PPL thấp ở đây.
    "scientific_technical": [
        "The process of cellular respiration involves a series of metabolic reactions and processes that take place in the cells of organisms to convert chemical energy into adenosine triphosphate (ATP).",
        "In quantum mechanics, the uncertainty principle, also known as Heisenberg's uncertainty principle, is any of a variety of mathematical inequalities asserting a fundamental limit to the precision with which the values for certain pairs of physical quantities of a particle, such as position and momentum, can be predicted from initial conditions.",
        "A neural network in machine learning is a computational model inspired by the structure and function of biological neural networks, consisting of interconnected nodes or neurons in a layered structure that resembles the human brain.",
        "The double-helix structure of DNA was first proposed by James Watson and Francis Crick in 1953, based on X-ray diffraction images taken by Rosalind Franklin.",
        "General relativity is a theory of gravitation developed by Albert Einstein between 1907 and 1915, which describes gravity as a geometric property of spacetime."
    ],
    
    # 3. Literary & Figurative Language
    # Thách thức model với ngôn ngữ ẩn dụ, văn chương, cấu trúc câu phức tạp.
    "literary_figurative": [
        "All the world's a stage, and all the men and women merely players; they have their exits and their entrances.",
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity.",
        "Hope is the thing with feathers that-perches in the soul, and sings the tune without the words, and never stops at all.",
        "The sun was a searing eye in the sky, and the pavement was a river of shimmering heat.",
        "Because I could not stop for Death, He kindly stopped for me; The Carriage held but just Ourselves and Immortality."
    ],

    # 4. Conversational & Informal Language
    # Kiểm tra model với ngôn ngữ đời thường, câu không hoàn chỉnh, tiếng lóng.
    "conversational_informal": [
        "Hey, what's up? You won't believe what happened to me at the supermarket today.",
        "So, like, I was thinking maybe we could grab a coffee sometime next week? Let me know what works for you.",
        "OMG, that movie was absolutely insane! The ending blew my mind, for real.",
        "Yeah, no worries. I'll get it done by tomorrow, promise. Just a bit swamped right now.",
        "Honestly, I'm not a big fan of that new restaurant. Kinda overrated, if you ask me."
    ],

    # 5. Programming Code (Python)
    # Rất quan trọng để đánh giá model cho các tác vụ liên quan đến lập trình.
    # Model có khả năng về code sẽ có PPL thấp hơn đáng kể ở đây.
    "code_python": [
        "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())",
        "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)",
        "class Node:\n    def __init__(self, data):\n        self.data = data\n        self.next = None",
        "try:\n    result = 10 / 0\nexcept ZeroDivisionError as e:\n    print(f'Error: Cannot divide by zero. Details: {e}')",
        "user_input = input('Enter your name: ')\nprint(f'Hello, {user_input}!')"
    ],
    
    # 6. Adversarial & Nonsensical
    # Các trường hợp "tiêu cực" để đảm bảo model không bị "ảo giác".
    # Model TỐT nên có PPL RẤT CAO ở đây.
    "adversarial_nonsense": [
        "Colorless green ideas sleep furiously.", # Câu đúng ngữ pháp nhưng vô nghĩa
        "The elephant whispered secrets to the moon using a telephone made of cheese.", # Câu có logic phi lý
        "runs dog the fast brown quick jumps lazy fox the over", # Từ đúng nhưng trật tự sai
        "asdf jkl; qweruiop zxcvbnm, lkjhgf poiu mnbv", # Văn bản ngẫu nhiên hoàn toàn
        "this sentence is not a sentence is not a this sentence is not.", # Lặp lại vô nghĩa
        "I are a programmer which writes codes goodly.", # Sai ngữ pháp nghiêm trọng
        "Le chat est sur la table, কিন্তু আমি বাংলা বলতে পারি, while mixing languages.", # Trộn lẫn nhiều ngôn ngữ
    ]
}

# Các tác vụ sinh văn bản truyền thống
TRADITIONAL_GENERATION_TASKS = {
    "Code Generation": [
        create_qwen_prompt("Viết một script Python sử dụng thư viện Pandas để đọc file CSV có tên `sales_data.csv` với các cột 'Date', 'Product', 'Revenue'. Script cần tính tổng doanh thu theo từng sản phẩm và xuất kết quả ra một file CSV mới có tên `revenue_by_product.csv`."),
        create_qwen_prompt("Tạo một component React functional bằng TypeScript tên là `UserProfile`. Component này nhận vào props là `name` (string), `age` (number), và `avatarUrl` (string), sau đó hiển thị thông tin này một cách có cấu trúc.")
    ],
    "Math Problem Solving": [
        create_qwen_prompt("Một bể nước có hai vòi. Vòi thứ nhất chảy một mình thì đầy bể trong 4 giờ. Vòi thứ hai chảy một mình thì đầy bể trong 6 giờ. Nếu mở cả hai vòi cùng một lúc khi bể cạn, hỏi sau bao lâu thì bể sẽ đầy? Trình bày các bước giải chi tiết."),
        create_qwen_prompt("Một người gửi tiết kiệm 500 triệu đồng với lãi suất kép 6.5% mỗi năm. Hỏi sau 5 năm, người đó sẽ nhận được cả vốn lẫn lãi là bao nhiêu tiền? Yêu cầu trình bày công thức và các bước tính toán.")
    ],
    "Text Summarization": [
        create_qwen_prompt("Hãy tóm tắt đoạn văn sau thành 3 ý chính: 'Các công nghệ thu giữ carbon (Carbon Capture Technologies) đang nổi lên như một giải pháp tiềm năng trong cuộc chiến chống biến đổi khí hậu. Các phương pháp chính bao gồm thu giữ sau đốt cháy, thu giữ trước đốt cháy và thu giữ từ không khí trực tiếp (DAC). Mặc dù có tiềm năng lớn, công nghệ này vẫn đối mặt với những thách thức về chi phí vận hành cao, hiệu quả năng lượng và vấn đề lưu trữ carbon an toàn trong dài hạn. Các chính phủ và tập đoàn lớn đang đầu tư hàng tỷ USD vào R&D để cải thiện hiệu quả và giảm giá thành, hy vọng biến nó thành một công cụ chủ chốt vào năm 2050.'"),
        create_qwen_prompt("Tóm tắt ngắn gọn những sự kiện chính và ý nghĩa lịch sử của cuộc Cách mạng Công nghiệp lần thứ nhất, tập trung vào các phát minh quan trọng và tác động của nó đến xã hội.")
    ],
    "Creative Writing": [
        create_qwen_prompt("Viết đoạn mở đầu cho một câu chuyện ngắn thuộc thể loại khoa học viễn tưởng, trong đó nhân vật chính là một nhà thực vật học sống trên Sao Hỏa, người vừa phát hiện ra một loài cây có khả năng giao tiếp bằng ánh sáng."),
        create_qwen_prompt("Viết một bài văn ngắn (khoảng 150 từ) miêu tả cảnh một phiên chợ nổi trên sông Cửu Long vào buổi sáng sớm, tập trung vào âm thanh, màu sắc và mùi vị đặc trưng.")
    ],
    "Question Answering": [
        create_qwen_prompt("Giải thích sự khác biệt cơ bản giữa năng lượng hạt nhân phân hạch (nuclear fission) và tổng hợp hạt nhân (nuclear fusion). Loại nào hiện đang được sử dụng trong các nhà máy điện?"),
        create_qwen_prompt("Con đường tơ lụa là gì và nó có vai trò quan trọng như thế nào đối với sự phát triển của các nền văn minh cổ đại?")
    ],
    "Language Translation & Nuance": [
        create_qwen_prompt("Dịch đoạn văn sau sang tiếng Việt, chú ý giữ văn phong chuyên nghiệp: 'Our team is conducting a comprehensive due diligence process to assess the viability of the potential acquisition. Key performance indicators (KPIs) and financial statements are under rigorous scrutiny.'"),
        create_qwen_prompt("Giải thích ý nghĩa và tìm câu thành ngữ tiếng Anh có nghĩa tương đương với câu 'Nước đến chân mới nhảy'.")
    ],
    "Reasoning & Logic": [
        create_qwen_prompt("Có ba chiếc hộp. Một hộp chứa toàn bi đỏ, một hộp chứa toàn bi xanh, và một hộp chứa lẫn lộn cả bi đỏ và bi xanh. Cả ba hộp đều bị dán nhãn sai. Bạn chỉ được phép lấy ra một viên bi từ một hộp duy nhất (không được nhìn vào bên trong). Làm thế nào để xác định chính xác nội dung của cả ba hộp? Hãy giải thích quá trình suy luận của bạn."),
        create_qwen_prompt("Một nghiên cứu cho thấy những thành phố có nhiều cửa hàng kem nhất cũng có tỷ lệ tội phạm cao nhất. Có phải ăn kem gây ra tội phạm không? Hãy giải thích về mối tương quan và quan hệ nhân quả (correlation vs. causation) trong trường hợp này.")
    ],
    "Technical Explanation": [
        create_qwen_prompt("Hãy giải thích cho một người không rành về công nghệ về khái niệm 'Điện toán đám mây' (Cloud Computing) là gì. Sử dụng ví dụ về Google Drive hoặc iCloud để minh họa."),
        create_qwen_prompt("Giải thích một cách đơn giản quá trình quang hợp ở thực vật diễn ra như thế nào và tại sao nó lại quan trọng đối với sự sống trên Trái Đất.")
    ]
}