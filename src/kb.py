"""Small plain-language knowledge base + TF-IDF retriever. Verify facts against official sources before real use."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB = [
 ("RTI Act 2005 - how to file", "Any citizen can request information from a public authority by writing to its Public Information Officer (PIO). A written application is addressed to the PIO of the concerned department. A small application fee applies (for central departments it is commonly Rs 10; states set their own fees). Verify on the official source."),
 ("RTI Act 2005 - time limits and appeals", "The PIO should normally reply within 30 days. If information concerns life or liberty it is due within 48 hours. If you get no reply or are unhappy, file a first appeal with the First Appellate Authority, generally within 30 days. A second appeal can go to the Information Commission, generally within 90 days. Verify on the official source."),
 ("Consumer Protection Act 2019 - what you can complain about", "A consumer can complain about defective goods, deficient services, unfair trade practices or overcharging. Keep the bill, warranty card, screenshots, emails and chats as evidence. First send a written complaint or legal notice to the seller or service provider and give a reasonable deadline to respond."),
 ("Consumer Protection Act 2019 - forums and limitation", "Complaints are generally filed within 2 years from the date the cause of action arose. Consumer Commissions exist at district, state and national levels depending on the value of goods or services; most ordinary claims start at the District Commission. Complaints can be filed online through the e-Daakhil portal. The National Consumer Helpline (1915) can help resolve disputes before filing. Verify on the official source."),
 ("Tenant rights - security deposit", "A security deposit is normally refundable at the end of the tenancy, after lawful deductions for unpaid rent or damage beyond normal wear and tear, as per the rent agreement. Rent laws differ by state, so check your state rent control act or tenancy rules. Keep the rent agreement, rent receipts, bank transfer proof, and proof of the date you vacated. Verify on the official source."),
 ("Tenant rights - eviction, entry and harassment", "A landlord cannot lawfully force a tenant out by cutting water or electricity or by entering without notice. Eviction generally requires a notice and a legal process as per the state tenancy law. A first step is a written demand or legal notice; if ignored, options include the rent authority, civil court or mediation. Verify on the official source."),
 ("Workplace rights - wages and termination", "Employers must pay wages on time as per labour law. Unpaid wages, unlawful deductions and non-payment of provident fund or final settlement can be raised first in writing with HR, then with the Labour Commissioner or labour office of your state. Keep your offer letter, payslips, bank statements and messages. Rules differ for different kinds of establishments. Verify on the official source."),
 ("Workplace rights - harassment (POSH)", "Under the Sexual Harassment of Women at Workplace Act 2013, workplaces with 10 or more employees must have an Internal Committee. A complaint is generally made in writing within 3 months of the incident (extendable in some cases). Verify on the official source."),
 ("Free legal help in India", "District Legal Services Authorities (DLSA) and NALSA provide free legal aid to eligible persons. Tele-Law connects citizens with lawyers through Common Service Centres. NALSA helpline: 15100. National Consumer Helpline: 1915. Emergency: 112. Women helpline: 181. Verify numbers on official websites before relying on them."),
]
_v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit([a + " " + b for a, b in KB])
_M = _v.transform([a + " " + b for a, b in KB])

def retrieve(query, k=3):
    s = cosine_similarity(_v.transform([query]), _M)[0]
    return [{"source": KB[i][0], "text": KB[i][1], "score": float(s[i])} for i in s.argsort()[::-1][:k] if s[i] > 0.02]
