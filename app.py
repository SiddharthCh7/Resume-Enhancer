import os, requests, json
import tempfile
import re
from flask import Flask, render_template, request, send_file, redirect, url_for, session
import fitz
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from dotenv import load_dotenv
load_dotenv()



app = Flask(__name__)
app.secret_key = 'resume_enhancer_secret_key'
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None


def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return None


def enhance_resume(resume_text):
    try:
        system_prompt = """You are a professional resume enhancement expert. 
            Your task is to improve the provided resume text while COMPLETELY PRESERVING ALL of the original content and sections.

            What You Should Do:
            Improve readability, conciseness, and clarity without changing the meaning.
            Make the writing more polished and professional while keeping it natural.
            
            What You Should NOT Do:
            Don't add anything new—not even for credibility.
            Don't change the order, structure, or formatting in any way.
            Don't remove anything, even if it seems unnecessary.
            Don't rewrite sentences in a way that alters the original intent.
            Your goal is to make the resume sound crisp, clear, and professional—without changing what's actually being said. Stick to the original content and just refine the language.



            Format your response with two clearly marked sections:
            <IMPROVED_RESUME>
            [The complete improved resume text with EVERY SINGLE element from the original preserved]
            </IMPROVED_RESUME>
            
            <CHANGES_MADE>
            [Bulleted list of specific language improvements made]
            </CHANGES_MADE>
            """

        user_message = f"Here is my resume text to enhance. You MUST keep ALL sections, ALL projects, and ALL technical skills:\n\n{resume_text}"
    
        payload = {
            "model": "deepseek/deepseek-r1:free",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        }
        headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}", 
        "Content-Type": "application/json"
        }
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions", 
            headers=headers, 
            data=json.dumps(payload)
        )
        
        result = response.json()["choices"][0]["message"]["content"]
        
        improved_resume_match = re.search(r"<IMPROVED_RESUME>\s*(.*?)\s*</IMPROVED_RESUME>", result, re.DOTALL)
        changes_made_match = re.search(r"<CHANGES_MADE>\s*(.*?)\s*</CHANGES_MADE>", result, re.DOTALL)


        improved_resume = improved_resume_match.group(1).strip() if improved_resume_match else "No improvements found."
        print("Improved Resume:")
        print(improved_resume)
        changes_made = changes_made_match.group(1).strip() if changes_made_match else "No changes specified."
        
        return {
            "improved_resume": improved_resume,
            "changes_made": changes_made,
            "error": False
        }
        
    except Exception as e:
        return {
            "error": True,
            "message": f"Error: {str(e)}"
        }




def write_to_pdf(text):
    
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], "enhanced_resume.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=40,
        bottomMargin=40
    )
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    centered_style = ParagraphStyle(
        'Centered',
        parent=styles['Normal'],
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        fontSize=14
    )
    
    bold_style = ParagraphStyle(
        'Bold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold'
    )
    
    paragraphs = []
    
    for line in text.split("\n"):
        if not line.strip():
            paragraphs.append(Paragraph("<br/>", normal_style))
            continue
            
        line = re.sub(r'\*\*\*\*\*\*(.*?)\*\*\*\*\*\*', r'<b>\1</b>', line)
        
        line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
        
        if re.match(r'^<b>.*</b>$', line) and len(line) < 60: 
            clean_line = re.sub(r'</?b>', '', line)
            paragraphs.append(Paragraph(clean_line, centered_style))
        else:
            paragraphs.append(Paragraph(line, normal_style))
    
    doc.build(paragraphs)
    
    print(f"PDF saved at {pdf_path}")
    doc = fitz.open(pdf_path)
    print("PDF Content:")
    for page in doc:
        print(page.get_text())
    doc.close()
    
    return pdf_path




def write_to_docx(text):
    print("well: ")
    print(text)
    docx_path = os.path.join(app.config['UPLOAD_FOLDER'], "enhanced_resume.docx")
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)

    doc.save(docx_path)
    print(f"DOCX saved at {docx_path}\n")

    doc = Document(docx_path)
    print("HERE IG:")
    for para in doc.paragraphs:
        print(para.text)

    return docx_path






@app.route('/')
def index():
    return render_template('index_flask.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'resume' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['resume']
    output_format = request.form.get('format', 'pdf')
    
    if file.filename == '':
        return redirect(url_for('index'))

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    if file.filename.endswith('.pdf'):
        resume_text = extract_text_from_pdf(file_path)
        original_format = 'pdf'
    elif file.filename.endswith('.docx'):
        resume_text = extract_text_from_docx(file_path)
        original_format = 'docx'
    else:
        return "Unsupported file format. Please upload a PDF or DOCX file."
    
    if not resume_text:
        return "Failed to extract text from the uploaded file."
    
    enhancement_result = enhance_resume(resume_text)
    session['improved_resume'] = enhancement_result['improved_resume']
    session['changes_made'] = enhancement_result['changes_made']
    session['output_format'] = output_format
    session['original_format'] = original_format
    
    return redirect(url_for('show_results'))


@app.route('/results')
def show_results():
    if 'improved_resume' not in session:
        return redirect(url_for('index'))
    
    changes_made = session.get('changes_made')
    formatted_changes = ""
    for line in changes_made.split('\n'):
        if line.strip():
            if line.strip().startswith('•') or line.strip().startswith('-') or line.strip().startswith('*'):
                formatted_changes += f"<li>{line.strip()[1:].strip()}</li>"
            else:
                formatted_changes += f"<li>{line.strip()}</li>"
    
    return render_template('results_flask.html', 
                          changes_made=formatted_changes)


@app.route('/download')
def download_file():
    if 'improved_resume' not in session:
        return redirect(url_for('index'))
    
    improved_resume = session.get('improved_resume')
    output_format = session.get('output_format', 'pdf')

    if output_format == 'docx':
        output_path = write_to_docx(improved_resume)
        mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        filename = "enhanced_resume.docx"
    else:
        output_path = write_to_pdf(improved_resume, )
        mime_type = 'application/pdf'
        filename = "enhanced_resume.pdf"
    if not os.path.exists(output_path) or not os.access(output_path, os.R_OK):
        return "Error: Could not generate the file. Please try again."
    
    return send_file(output_path, as_attachment=True, 
                    download_name=filename, mimetype=mime_type)


if __name__ == '__main__':
    app.run(debug=True)