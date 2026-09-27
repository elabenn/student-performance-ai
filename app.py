from ucimlrepo import fetch_ucirepo
import streamlit as st
import joblib
import pandas as pd


# Sayfa ayarları
st.set_page_config(
    page_title="Student Performance AI",
    layout="centered"
)

# Eğitilmiş modeli yükle
model = joblib.load("student_performance_model.pkl")


@st.cache_data
def load_data():
    student_performance = fetch_ucirepo(id=320)
    return (
        student_performance.data.features,
        student_performance.data.targets
    )


X, y = load_data()


st.title("Student Performance AI")

st.write(
    "Öğrenci özelliklerine göre final notunu (G3) "
    "tahmin eden makine öğrenmesi uygulaması."
)

st.divider()

st.subheader("Öğrenci Bilgileri")

age = st.number_input(
    "Yaş",
    min_value=15,
    max_value=25,
    value=17
)

studytime = st.selectbox(
    "Haftalık çalışma süresi",
    [1, 2, 3, 4]
)

failures = st.selectbox(
    "Önceki başarısızlık sayısı",
    [0, 1, 2, 3]
)

absences = st.number_input(
    "Devamsızlık sayısı",
    min_value=0,
    max_value=100,
    value=5
)

traveltime = st.selectbox(
    "Okula ulaşım süresi",
    [1, 2, 3, 4]
)

schoolsup = st.selectbox(
    "Okuldan ekstra akademik destek alıyor musun?",
    ["no", "yes"]
)

higher = st.selectbox(
    "Üniversite eğitimi almak istiyor musun?",
    ["yes", "no"]
)

Medu = st.selectbox(
    "Anne eğitim düzeyi",
    [0, 1, 2, 3, 4],
    format_func=lambda x: {
        0: "Eğitim yok",
        1: "İlkokul",
        2: "Ortaokul",
        3: "Lise",
        4: "Yükseköğrenim"
    }[x]
)

Fedu = st.selectbox(
    "Baba eğitim düzeyi",
    [0, 1, 2, 3, 4],
    format_func=lambda x: {
        0: "Eğitim yok",
        1: "İlkokul",
        2: "Ortaokul",
        3: "Lise",
        4: "Yükseköğrenim"
    }[x]
)

famsup = st.selectbox(
    "Aileden eğitim desteği alıyor musun?",
    ["yes", "no"]
)

freetime = st.selectbox(
    "Boş zamanını değerlendirme sıklığın",
    [1, 2, 3, 4, 5]
)

goout = st.selectbox(
    "Arkadaşlarınla dışarı çıkma sıklığın",
    [1, 2, 3, 4, 5]
)

health = st.selectbox(
    "Genel sağlık durumun",
    [1, 2, 3, 4, 5]
)

activities = st.selectbox(
    "Okul dışı aktivitelere katılıyor musun?",
    ["yes", "no"]
)

romantic = st.selectbox(
    "Romantik bir ilişkin var mı?",
    ["yes", "no"]
)

internet = st.selectbox(
    "Evde internet erişimin var mı?",
    ["yes", "no"]
)

school = st.selectbox(
    "Okul türü",
    ["Devlet okulu", "Meslek okulu"]
)

school = "GP" if school == "Devlet okulu" else "MS"

sex = st.selectbox(
    "Cinsiyet",
    ["Kadın", "Erkek"]
)

sex = "F" if sex == "Kadın" else "M"

address = st.selectbox(
    "Yaşam yeri",
    ["Şehir", "Kırsal"]
)

address = "U" if address == "Şehir" else "R"

famsize = st.selectbox(
    "Ailedeki kişi sayısı",
    ["3'ten fazla", "3 veya daha az"]
)

famsize = "GT3" if famsize == "3'ten fazla" else "LE3"

Pstatus = st.selectbox(
    "Ebeveynlerin birlikte yaşama durumu",
    ["Birlikte", "Ayrı"]
)

Pstatus = "T" if Pstatus == "Birlikte" else "A"

famrel = st.selectbox(
    "Aile ilişkilerinin yakınlığı",
    [1, 2, 3, 4, 5]
)

Mjob = st.selectbox(
    "Anne mesleği",
    ["Öğretmen", "Sağlık sektörü", "Hizmet sektörü", "Evde", "Diğer"]
)

Mjob = {
    "Öğretmen": "teacher",
    "Sağlık sektörü": "health",
    "Hizmet sektörü": "services",
    "Evde": "at_home",
    "Diğer": "other"
}[Mjob]

Fjob = st.selectbox(
    "Baba mesleği",
    ["Öğretmen", "Sağlık sektörü", "Hizmet sektörü", "Evde", "Diğer"]
)

Fjob = {
    "Öğretmen": "teacher",
    "Sağlık sektörü": "health",
    "Hizmet sektörü": "services",
    "Evde": "at_home",
    "Diğer": "other"
}[Fjob]

reason = st.selectbox(
    "Okul seçme nedeni",
    [
        "Eve yakın olması",
        "Okulun itibarı",
        "Ders/program seçenekleri",
        "Diğer"
    ]
)

reason = {
    "Eve yakın olması": "home",
    "Okulun itibarı": "reputation",
    "Ders/program seçenekleri": "course",
    "Diğer": "other"
}[reason]

guardian = st.selectbox(
    "Yasal vasi",
    ["Anne", "Baba", "Diğer"]
)

guardian = {
    "Anne": "mother",
    "Baba": "father",
    "Diğer": "other"
}[guardian]


if st.button("Final Notunu Tahmin Et"):

    student_data = pd.DataFrame([{
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": "no",
        "activities": activities,
        "nursery": "yes",
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": 1,
        "Walc": 1,
        "health": health,
        "absences": absences
    }])

    student_data = pd.get_dummies(
        student_data,
        drop_first=True
    )

    student_data = student_data.reindex(
        columns=model.feature_names_in_,
        fill_value=False
    )

    prediction = model.predict(student_data)[0]

    st.subheader("Tahmin Sonucu")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Tahmini final notu",
            f"{prediction:.1f} / 20"
        )

    with col2:
        st.metric(
            "Modelin ortalama hatası",
            "≈ 2.1 puan"
        )

    if prediction >= 15:
        st.success(
            "Model tahmini yüksek bir final notuna işaret ediyor."
        )
    elif prediction >= 10:
        st.warning(
            "Model tahmini orta seviyede bir final notuna işaret ediyor."
        )
    else:
        st.error(
            "Model tahmini düşük bir final notuna işaret ediyor."
        )

    st.info(
        "Bu sonuç, makine öğrenmesi modelinin öğrencinin "
        "girdiği özelliklere göre yaptığı tahmindir. "
        "Gerçek final notu farklı olabilir."
    )


st.divider()

st.subheader("Model Hakkında")

st.write(
    "Bu uygulama, öğrencilerin çeşitli akademik ve sosyal özelliklerini "
    "kullanarak final notunu (G3) tahmin etmek için Random Forest "
    "regresyon modeli kullanır."
)

st.write(
    "Model, UCI Student Performance veri setindeki 649 öğrenci "
    "üzerinde geliştirilmiştir."
)

st.write(
    "Test verisinde modelin ortalama mutlak hatası yaklaşık 2.1 puandır "
    "ve R² skoru yaklaşık 0.18'dir."
)


st.divider()

st.subheader("Veri Seti Analizi")

st.write(
    "Aşağıdaki grafik, UCI Student Performance veri setindeki "
    "final notlarının (G3) dağılımını göstermektedir."
)

st.bar_chart(
    y["G3"].value_counts().sort_index()
)


st.subheader("Çalışma Süresi ve Final Notu")

studytime_data = X.copy()
studytime_data["G3"] = y["G3"]

st.scatter_chart(
    studytime_data,
    x="studytime",
    y="G3"
)


st.subheader("Başarısızlık Sayısı ve Final Notu")

failures_data = X.copy()
failures_data["G3"] = y["G3"]

st.scatter_chart(
    failures_data,
    x="failures",
    y="G3"
)


st.divider()

st.subheader("Model Performansı")

st.write(
    "Random Forest modelinin test verisi üzerindeki performansı "
    "aşağıdaki metriklerle değerlendirilmiştir."
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "MAE",
        "2.06 puan"
    )

with col2:
    st.metric(
        "R² Skoru",
        "0.18"
    )

st.info(
    "MAE, tahminlerin gerçek final notlarından ortalama uzaklığını "
    "gösterir. R² ise modelin test verisindeki değişkenliğin ne "
    "kadarını açıkladığını gösterir."
)


st.subheader("Model Karşılaştırması")

comparison_data = pd.DataFrame(
    {
        "Model": [
            "Random Forest",
            "TensorFlow",
            "Geliştirilmiş TensorFlow"
        ],
        "R² Skoru": [
            0.1802,
            0.1082,
            0.0855
        ]
    }
)

st.bar_chart(
    comparison_data.set_index("Model")
)

st.write(
    "Bu karşılaştırma, üç modelin test verisindeki R² skorlarını "
    "göstermektedir. R² değeri modelin test verisindeki açıklama "
    "gücünü ifade eder."
)