"""
Module de génération de PDF pour les rapports de rénovation énergétique.
Utilise FPDF2 pour créer des PDFs propres et cohérents.
"""

import os
import re
import unicodedata
from datetime import datetime
from fpdf import FPDF


def sanitize_text(text):
    """
    Nettoie le texte pour être compatible avec les polices standard PDF.
    """
    if not text:
        return ""
    
    text = str(text)
    
    # Nettoyer les balises markdown ** et * en premier
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **texte** -> texte
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *texte* -> texte
    text = re.sub(r'\*\*+', '', text)               # ** restants
    text = re.sub(r'\*+', '', text)                  # * restants
    
    replacements = {
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'à': 'a', 'â': 'a', 'ä': 'a',
        'ù': 'u', 'û': 'u', 'ü': 'u',
        'ô': 'o', 'ö': 'o', 'ò': 'o',
        'î': 'i', 'ï': 'i', 'ì': 'i',
        'ç': 'c',
        'É': 'E', 'È': 'E', 'Ê': 'E', 'Ë': 'E',
        'À': 'A', 'Â': 'A', 'Ä': 'A',
        'Ù': 'U', 'Û': 'U', 'Ü': 'U',
        'Ô': 'O', 'Ö': 'O', 'Ò': 'O',
        'Î': 'I', 'Ï': 'I', 'Ì': 'I',
        'Ç': 'C',
        '°': ' deg',
        '²': '2',
        '€': ' EUR',
        '–': '-',
        '—': '-',
        ''': "'",
        ''': "'",
        '"': '"',
        '"': '"',
        '…': '...',
        '\r\n': '\n',
        '\r': '\n',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    result = ""
    for char in text:
        if ord(char) < 128:
            result += char
        else:
            try:
                normalized = unicodedata.normalize('NFKD', char)
                ascii_char = normalized.encode('ascii', 'ignore').decode('ascii')
                result += ascii_char if ascii_char else ' '
            except:
                result += ' '
    
    return result


def parse_building_info(prompt_text):
    """Extrait les informations du bâtiment depuis le prompt RAG."""
    info = {
        'dpe_actuel': 'N/A',
        'departement': 'N/A',
        'annee': 'N/A',
        'surface': 'N/A',
        'ubat': 'N/A',
        'conso_chauffage': 'N/A',
        'emissions_co2': 'N/A',
        'diagnostic': 'N/A'
    }
    
    match = re.search(r'DPE ACTUEL\s*:\s*([A-G])', prompt_text, re.IGNORECASE)
    if match:
        info['dpe_actuel'] = match.group(1)
    
    match = re.search(r'département\s*(\d+).*Année\s*(\d+).*?(\d+\.?\d*)\s*m', prompt_text, re.IGNORECASE)
    if match:
        info['departement'] = match.group(1)
        info['annee'] = match.group(2)
        info['surface'] = f"{match.group(3)} m2"
    
    match = re.search(r'Isolation.*?:\s*([\d.]+)\s*W/m', prompt_text, re.IGNORECASE)
    if match:
        info['ubat'] = f"{match.group(1)} W/m2.K"
    
    match = re.search(r'Conso Chauffage\s*:\s*(\d+)\s*kWhEP/m', prompt_text, re.IGNORECASE)
    if match:
        info['conso_chauffage'] = f"{match.group(1)} kWhEP/m2"
    
    match = re.search(r'Emissions CO2.*?:\s*(\d+)\s*kgCO2/m', prompt_text, re.IGNORECASE)
    if match:
        info['emissions_co2'] = f"{match.group(1)} kgCO2/m2"
    
    match = re.search(r'DIAGNOSTIC\s*:\s*(.+?)(?:TACHE|$)', prompt_text, re.IGNORECASE | re.DOTALL)
    if match:
        diag = match.group(1).strip()
        diag = re.sub(r'\s+', ' ', diag)
        info['diagnostic'] = diag[:250] + '...' if len(diag) > 250 else diag
    
    return info


def _extract_section(text, start_tag, end_tag, next_section_pattern=None):
    """
    Extrait une section entre balises, avec fallback sur la section suivante.
    """
    # Essayer avec balise de fermeture
    pattern = rf'\[{start_tag}\](.*?)\[/{start_tag}\]'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: jusqu'à la prochaine section ou fin de texte
    pattern = rf'\[{start_tag}\](.*?)(?=\[(?:SCENARIO_|ANALYSE|/)|$)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return ""


def parse_rag_response(response_text):
    """Parse la réponse structurée du RAG avec support pour balises manquantes."""
    result = {
        'analyse': '',
        'scenario_1': {
            'titre': 'Travaux Legers - Isolation',
            'description': '',
            'classe_visee': 'N/A',
            'cout_estime': 'N/A',
            'aide_montant': 'N/A',
            'economies_annuelles': 'N/A',
            'rentabilite': 'N/A'
        },
        'scenario_2': {
            'titre': 'Travaux Complets - Renovation Globale',
            'description': '',
            'classe_visee': 'N/A',
            'cout_estime': 'N/A',
            'aide_montant': 'N/A',
            'economies_annuelles': 'N/A',
            'rentabilite': 'N/A'
        }
    }
    
    # Analyse
    result['analyse'] = _extract_section(response_text, 'ANALYSE', '/ANALYSE')
    
    # Scénario 1
    s1_text = _extract_section(response_text, 'SCENARIO_1', '/SCENARIO_1')
    if s1_text:
        _parse_scenario(s1_text, result['scenario_1'], 'Aide_CEE')
    
    # Scénario 2
    s2_text = _extract_section(response_text, 'SCENARIO_2', '/SCENARIO_2')
    if s2_text:
        _parse_scenario(s2_text, result['scenario_2'], 'Aide_MaPrimeRenov')
    
    # Fallback si pas de structure
    if not result['analyse'] and not result['scenario_1']['description']:
        result['analyse'] = response_text[:800] if len(response_text) > 800 else response_text
    
    return result


def _parse_scenario(scenario_text, scenario_dict, aide_key):
    """Parse un scénario individuel avec tous les nouveaux champs."""
    
    # Nettoyer TOUTES les balises markdown ** (même imbriquées ou multiples)
    # Remplacer **texte** par texte (gras)
    clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', scenario_text)
    # Remplacer *texte* par texte (italique)
    clean_text = re.sub(r'\*([^*]+)\*', r'\1', clean_text)
    # Nettoyer les ** restants (cas où il y a des ** isolés)
    clean_text = re.sub(r'\*\*+', '', clean_text)
    clean_text = re.sub(r'\*+', '', clean_text)
    
    # Titre
    match = re.search(r'Titre:\s*(.+?)(?:\n|$)', clean_text)
    if match:
        scenario_dict['titre'] = match.group(1).strip()
    
    # Description - capture jusqu'à Classe_Visee ou Classe energetique visee
    match = re.search(r'Description:\s*(.+?)(?=Classe[_\s]*(?:Visee|energetique[_\s]*visee):|$)', clean_text, re.DOTALL | re.IGNORECASE)
    if match:
        desc = match.group(1).strip()
        # Nettoyer les balises markdown ** et *
        desc = re.sub(r'\*\*([^*]+)\*\*', r'\1', desc)
        desc = re.sub(r'\*([^*]+)\*', r'\1', desc)
        desc = re.sub(r'\*\*+', '', desc)
        desc = re.sub(r'\*+', '', desc)
        # Nettoyer les sauts de ligne multiples et les bullets markdown
        desc = re.sub(r'\n\s*\n', '\n', desc)
        desc = re.sub(r'^\s*\*\s*', '- ', desc, flags=re.MULTILINE)
        scenario_dict['description'] = desc
    
    # Classe visée - patterns plus flexibles
    match = re.search(r'Classe[_\s]*Visee[_\s]*:\s*\[?([A-G])\]?', clean_text, re.IGNORECASE)
    if match:
        scenario_dict['classe_visee'] = match.group(1).upper()
    else:
        # Essayer avec "Classe energetique visee"
        match = re.search(r'Classe[_\s]*energetique[_\s]*visee[_\s]*:\s*\[?([A-G])\]?', clean_text, re.IGNORECASE)
        if match:
            scenario_dict['classe_visee'] = match.group(1).upper()
    
    # Coût estimé - patterns plus flexibles
    match = re.search(r'Cout[_\s]*Estime[_\s]*:\s*\[?([^\]\n]+)\]?', clean_text, re.IGNORECASE)
    if match:
        cout = match.group(1).strip()
        # Nettoyer les balises markdown restantes
        cout = re.sub(r'\*\*+', '', cout)
        cout = re.sub(r'\*+', '', cout)
        scenario_dict['cout_estime'] = cout
    
    # Aide financière - patterns plus flexibles
    match = re.search(rf'{aide_key}[_\s]*:\s*\[?([^\]\n]+)\]?', clean_text, re.IGNORECASE)
    if match:
        aide = match.group(1).strip()
        # Nettoyer les balises markdown restantes
        aide = re.sub(r'\*\*+', '', aide)
        aide = re.sub(r'\*+', '', aide)
        scenario_dict['aide_montant'] = aide
    
    # Économies annuelles - patterns plus flexibles
    match = re.search(r'Economies[_\s]*Annuelles[_\s]*:\s*\[?([^\]\n]+)\]?', clean_text, re.IGNORECASE)
    if match:
        econ = match.group(1).strip()
        # Nettoyer les balises markdown restantes
        econ = re.sub(r'\*\*+', '', econ)
        econ = re.sub(r'\*+', '', econ)
        scenario_dict['economies_annuelles'] = econ
    
    # Rentabilité - patterns plus flexibles
    match = re.search(r'Rentabilite[_\s]*:\s*\[?([^\]\n]+)\]?', clean_text, re.IGNORECASE)
    if match:
        rent = match.group(1).strip()
        # Nettoyer les balises markdown restantes
        rent = re.sub(r'\*\*+', '', rent)
        rent = re.sub(r'\*+', '', rent)
        scenario_dict['rentabilite'] = rent


def generate_renovation_pdf(building_info, parsed_response, output_path):
    """Génère le PDF du rapport de rénovation."""
    # Couleur principale GreenDiag - vert herbe
    GREEN_DIAG = (76, 140, 43)
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Logo GreenDiag
    logo_path = "./greendiag.png"
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=10, y=8, w=25)
    
    # En-tête avec titre GreenDiag
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(*GREEN_DIAG)
    pdf.cell(0, 10, sanitize_text('Rapport de Renovation Energetique'), align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 6, sanitize_text('GreenDiag'), align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(2)
    pdf.set_draw_color(*GREEN_DIAG)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)
    
    # Section 1: Description du logement
    _add_section_title(pdf, 'Description du Logement', (52, 73, 94))
    
    _add_field_row(pdf, 'Classe DPE actuelle', building_info['dpe_actuel'])
    _add_field_row(pdf, 'Departement', building_info['departement'])
    _add_field_row(pdf, 'Annee de construction', building_info['annee'])
    _add_field_row(pdf, 'Surface', building_info['surface'])
    pdf.ln(2)
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, sanitize_text('Donnees techniques:'), new_x='LMARGIN', new_y='NEXT')
    _add_field_row(pdf, 'Isolation (Ubat)', building_info['ubat'])
    _add_field_row(pdf, 'Consommation chauffage', building_info['conso_chauffage'])
    _add_field_row(pdf, 'Emissions CO2', building_info['emissions_co2'])
    pdf.ln(2)
    
    if building_info['diagnostic'] != 'N/A':
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 6, sanitize_text('Diagnostic:'), new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5, sanitize_text(building_info['diagnostic']))
        pdf.set_text_color(0, 0, 0)
    
    pdf.ln(4)
    
    # Section 2: Analyse
    if parsed_response['analyse']:
        _add_section_title(pdf, 'Analyse Energetique', (142, 68, 173))
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5, sanitize_text(parsed_response['analyse']))
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)
    
    # Scénario 1 - Vert herbe GreenDiag
    _add_scenario_section(pdf, 1, parsed_response['scenario_1'], 'CEE', (76, 140, 43))
    
    # Scénario 2 - Vert herbe GreenDiag (légèrement plus foncé)
    _add_scenario_section(pdf, 2, parsed_response['scenario_2'], 'MaPrimeRenov', (56, 120, 33))
    
    # Footer
    pdf.ln(5)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(128, 128, 128)
    date_str = datetime.now().strftime('%d/%m/%Y')
    pdf.cell(0, 10, sanitize_text(f'Rapport genere le {date_str}'), align='C')
    
    # Sauvegarde
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pdf.output(output_path)
    print(f"PDF genere avec succes: {output_path}")
    return output_path


def _add_section_title(pdf, title, color):
    """Ajoute un titre de section."""
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(*color)
    pdf.cell(0, 8, sanitize_text(title), new_x='LMARGIN', new_y='NEXT')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)


def _add_field_row(pdf, label, value):
    """Ajoute une ligne label: valeur."""
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(55, 5, sanitize_text(f'{label}:'))
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 5, sanitize_text(str(value)), new_x='LMARGIN', new_y='NEXT')


def _add_scenario_section(pdf, num, scenario, aide_type, color):
    """Ajoute une section scénario complète."""
    # Titre du scénario
    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 11)
    title = f"Scenario {num}: {scenario['titre']}"
    pdf.cell(0, 8, sanitize_text(title), fill=True, new_x='LMARGIN', new_y='NEXT')
    
    pdf.ln(2)
    pdf.set_text_color(0, 0, 0)
    
    # Description
    if scenario['description']:
        pdf.set_font('Helvetica', 'B', 9)
        pdf.cell(0, 5, sanitize_text('Description des travaux:'), new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 4, sanitize_text(scenario['description']))
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)
    
    # Tableau des résultats
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(240, 240, 240)
    
    # Classe visée
    pdf.cell(55, 6, sanitize_text('Classe energetique visee:'), border=1, fill=True)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(39, 174, 96)
    pdf.cell(0, 6, sanitize_text(f'  {scenario["classe_visee"]}'), border=1, new_x='LMARGIN', new_y='NEXT')
    pdf.set_text_color(0, 0, 0)
    
    # Coût estimé
    pdf.set_font('Helvetica', 'B', 9)
    pdf.cell(55, 6, sanitize_text('Cout estime:'), border=1, fill=True)
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(0, 6, sanitize_text(f'  {scenario["cout_estime"]}'), border=1, new_x='LMARGIN', new_y='NEXT')
    
    # Aide financière
    pdf.set_font('Helvetica', 'B', 9)
    pdf.cell(55, 6, sanitize_text(f'Aide {aide_type}:'), border=1, fill=True)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(0, 6, sanitize_text(f'  {scenario["aide_montant"]}'), border=1, new_x='LMARGIN', new_y='NEXT')
    pdf.set_text_color(0, 0, 0)
    
    # Économies annuelles
    pdf.set_font('Helvetica', 'B', 9)
    pdf.cell(55, 6, sanitize_text('Economies annuelles:'), border=1, fill=True)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(39, 174, 96)
    pdf.cell(0, 6, sanitize_text(f'  {scenario["economies_annuelles"]}'), border=1, new_x='LMARGIN', new_y='NEXT')
    pdf.set_text_color(0, 0, 0)
    
    # Rentabilité
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(255, 243, 205)  # Jaune clair pour mettre en évidence
    pdf.cell(55, 6, sanitize_text('Retour sur investissement:'), border=1, fill=True)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(180, 120, 0)
    pdf.cell(0, 6, sanitize_text(f'  {scenario["rentabilite"]}'), border=1, new_x='LMARGIN', new_y='NEXT')
    pdf.set_text_color(0, 0, 0)
    
    pdf.ln(6)


def generate_pdf_from_files(prompt_path, response_path, output_path):
    """Génère un PDF depuis les fichiers existants."""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_text = f.read()
    
    with open(response_path, 'r', encoding='utf-8') as f:
        response_text = f.read()
    
    building_info = parse_building_info(prompt_text)
    parsed_response = parse_rag_response(response_text)
    
    return generate_renovation_pdf(building_info, parsed_response, output_path)


if __name__ == "__main__":
    generate_pdf_from_files(
        "./prompts/prompts_rag.txt",
        "./outputs/reponse_rag.txt",
        "./outputs/rapport_renovation.pdf"
    )
