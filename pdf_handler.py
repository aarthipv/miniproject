#!/usr/bin/env python3
"""
PDF Handler for Latin Text Reconstruction Results
Generates professional PDF reports of text reconstruction results
"""

import os
import tempfile
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import logging

class PDFHandler:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the PDF."""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        # Subtitle style
        self.subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.darkgreen
        )
        
        # Section style
        self.section_style = ParagraphStyle(
            'CustomSection',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkred
        )
        
        # Text style
        self.text_style = ParagraphStyle(
            'CustomText',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        )
        
        # Code style for reconstructed text
        self.code_style = ParagraphStyle(
            'CustomCode',
            parent=self.styles['Code'],
            fontSize=11,
            spaceAfter=8,
            fontName='Courier',
            backColor=colors.lightgrey,
            borderWidth=1,
            borderColor=colors.grey,
            borderPadding=8
        )
        
        # Highlight style for reconstructed portions
        self.highlight_style = ParagraphStyle(
            'CustomHighlight',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            textColor=colors.darkblue,
            backColor=colors.lightyellow
        )

    def generate_pdf(self, original_text, reconstructed_text, highlighted_text, translation, filename=None):
        """
        Generate a PDF report of the text reconstruction results.
        
        Args:
            original_text (str): The original damaged text
            reconstructed_text (str): The reconstructed text
            highlighted_text (str): Text with highlighted reconstructions
            translation (str): English translation
            filename (str): Optional filename for the PDF
            
        Returns:
            str: Path to the generated PDF file
        """
        try:
            # Create temporary file if no filename provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"latin_reconstruction_{timestamp}.pdf"
            
            # Create PDF document
            doc = SimpleDocTemplate(filename, pagesize=A4)
            story = []
            
            # Add title
            title = Paragraph("Historical Text Reconstruction Report", self.title_style)
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Add timestamp
            timestamp_text = f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
            timestamp_para = Paragraph(timestamp_text, self.text_style)
            story.append(timestamp_para)
            story.append(Spacer(1, 30))
            
            # Add original text section
            story.append(Paragraph("Original Damaged Text", self.section_style))
            original_para = Paragraph(f'<b>Input:</b> {original_text}', self.code_style)
            story.append(original_para)
            story.append(Spacer(1, 15))
            
            # Add reconstructed text section
            story.append(Paragraph("Reconstructed Text", self.section_style))
            reconstructed_para = Paragraph(f'<b>Output:</b> {reconstructed_text}', self.code_style)
            story.append(reconstructed_para)
            story.append(Spacer(1, 15))
            
            # Add highlighted differences section
            story.append(Paragraph("Highlighted Reconstructions", self.section_style))
            highlighted_para = Paragraph(f'<b>Analysis:</b> {highlighted_text}', self.highlight_style)
            story.append(highlighted_para)
            story.append(Spacer(1, 15))
            
            # Add translation section
            story.append(Paragraph("English Translation", self.section_style))
            translation_para = Paragraph(f'<b>Translation:</b> {translation}', self.text_style)
            story.append(translation_para)
            story.append(Spacer(1, 20))
            
            # Add analysis table
            story.append(Paragraph("Reconstruction Analysis", self.section_style))
            analysis_data = [
                ['Metric', 'Value'],
                ['Original Length', f'{len(original_text)} characters'],
                ['Reconstructed Length', f'{len(reconstructed_text)} characters'],
                ['Characters Added', f'{len(reconstructed_text) - len(original_text)}'],
                ['Reconstruction Date', datetime.now().strftime('%Y-%m-%d')],
                ['Model Used', 'BART-based Latin Text Reconstruction']
            ]
            
            analysis_table = Table(analysis_data, colWidths=[2*inch, 3*inch])
            analysis_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(analysis_table)
            story.append(Spacer(1, 20))
            
            # Add footer
            footer_text = """
            <b>Generated by:</b> Historical Text Reconstruction System<br/>
            <b>Technology:</b> AI-powered BART model for Latin/Cyrillic text reconstruction<br/>
            <b>Translation:</b> Powered by Google Gemini AI
            """
            footer_para = Paragraph(footer_text, self.text_style)
            story.append(footer_para)
            
            # Build PDF
            doc.build(story)
            logging.info(f"PDF generated successfully: {filename}")
            return filename
            
        except Exception as e:
            logging.error(f"Error generating PDF: {e}")
            raise

    def generate_simple_pdf(self, original_text, reconstructed_text, translation):
        """
        Generate a simplified PDF with just the essential information.
        
        Args:
            original_text (str): The original damaged text
            reconstructed_text (str): The reconstructed text
            translation (str): English translation
            
        Returns:
            str: Path to the generated PDF file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_reconstruction_{timestamp}.pdf"
        
        doc = SimpleDocTemplate(filename, pagesize=letter)
        story = []
        
        # Title
        story.append(Paragraph("Text Reconstruction Result", self.title_style))
        story.append(Spacer(1, 20))
        
        # Content
        story.append(Paragraph("Original Text:", self.section_style))
        story.append(Paragraph(original_text, self.code_style))
        story.append(Spacer(1, 15))
        
        story.append(Paragraph("Reconstructed Text:", self.section_style))
        story.append(Paragraph(reconstructed_text, self.code_style))
        story.append(Spacer(1, 15))
        
        story.append(Paragraph("Translation:", self.section_style))
        story.append(Paragraph(translation, self.text_style))
        
        doc.build(story)
        return filename 