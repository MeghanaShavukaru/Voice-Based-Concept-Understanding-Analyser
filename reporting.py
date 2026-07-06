from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def generate_pdf_report(report_path, transcript, semantics, pause, filler, rms, final_score, feedback, waveform_img):
    c = canvas.Canvas(report_path, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "VBCUA: Concept Understanding Report")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Final Comprehension Score: {final_score:.2f} / 100")
    c.drawString(50, height - 100, f"Feedback: {feedback}")
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 130, "1. Semantic Metrics")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 150, f"Semantic Similarity: {semantics:.2f}%")
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 180, "2. Fluency Metrics")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 200, f"Filler Word Ratio: {filler:.2f}%")
    c.drawString(50, height - 220, f"Pause Ratio: {pause:.2f}%")
    c.drawString(50, height - 240, f"Average RMS Energy: {rms:.4f}")
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 270, "3. Transcription Summary")
    c.setFont("Helvetica", 10)
    # Basic text wrap for PDF
    textobject = c.beginText(50, height - 290)
    textobject.textLines(transcript[:800] + ("..." if len(transcript) > 800 else ""))
    c.drawText(textobject)
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 420, "4. Waveform Visualization")
    if waveform_img:
        img = ImageReader(waveform_img)
        c.drawImage(img, 50, height - 600, width=450, height=150)
        
    c.save()