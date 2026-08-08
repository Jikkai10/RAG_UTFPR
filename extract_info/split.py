
import re


ART_RE   = re.compile(r"Art(?:igo)?\.\s*(\d+)[ºo]?\b.*", re.IGNORECASE)


art = """
Art. 22 - A critério da coordenação do curso, a DIRGRAD poderá ofertar unidades curriculares com turmas que tenham vagas destinadas a estudantes sem presença obrigatória.

§ 1º Entende-se por estudantes que podem cursar unidades curriculares sem presença obrigatória aqueles que tenham sido reprovados na unidade curricular com nota final igual ou superior a 4,0 (quatro) e com frequência/participação mínima de 75% (setenta e cinco por cento).

§ 2º As unidades curriculares que poderão ter turmas com vagas destinadas a estudantes sem presença obrigatória deverão ser definidas pelo colegiado do curso.

§ 3º Os estudantes sem presença obrigatória deverão fazer todas as avaliações previstas para a turma da unidade curricular em que estão matriculados, conforme previsto no Planejamento de Aulas.

§ 4º Em caso de sobreposição de datas e horários entre avaliações presenciais das turmas nas quais esteja matriculado, o estudante terá direito a avaliação em segunda chamada conforme o Art. 37.

§ 5º Poderão ser ofertadas turmas nas quais todas as vagas destinam-se a estudantes sem presença obrigatória, desde que seja ofertada no mesmo dia e horário, uma turma presencial.

§ 6º As turmas em que todas as vagas são destinadas a estudantes sem presença obrigatória possuirão um Planejamento de Aulas específico, devendo ser presenciais as avaliações previstas neste planejamento.

§ 7º As turmas em que todas as vagas são destinadas a estudantes sem presença obrigatória poderão utilizar tecnologias de informação e comunicação e serão acompanhadas pela coordenação do curso.

§ 8º As cargas horárias das unidades curriculares cursadas na condição de estudante sem presença obrigatória serão computadas, conforme o que determina o § 2º do Art. 21.

§ 9º As unidades curriculares cursadas na condição de estudante sem presença obrigatória não serão consideradas para o que determina o § 5º do Art. 21.

§ 10. O preenchimento das turmas com vagas para estudantes sem presença obrigatória, seguirá os critérios estabelecidos no Art. 24.

§ 11. A cada período letivo, o estudante poderá matricular-se em, no máximo, 2 (duas) unidades curriculares na condição de estudante sem presença obrigatória.

§ 12. Nas turmas com vagas destinadas a estudantes sem presença obrigatória, no momento da matrícula, o estudante deverá optar por cursar a unidade curricular na condição de estudante sem presença obrigatória, desde que cumpra o § 1º deste artigo.
"""

matches = ART_RE.finditer(art)
for match in matches:
    print("Matched article number:", match.group(1))
