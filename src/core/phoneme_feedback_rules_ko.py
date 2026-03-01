from Levenshtein import editops

# 오류 유형별 피드백 정의
VARIANTS_FEEDBACK = {
    ('ㅃ', 'ㅂ'): (
        'ㅃ은 힘을 주어 두 입술을 단단히 붙였다가 떼면서 나는 소리입니다. 힘을 약하게 주면 ㅂ처럼 들릴 수 있으니 조심하세요.',
        'ㅃ: This sound is made by firmly pressing both lips together and releasing them with force. If you apply less pressure, it may sound like ㅂ, so be careful.'
    ),
    ('ㅉ', 'ㅈ'): (
        'ㅉ은 혀를 입천장에 단단히 붙였다가 떼면서 나는 소리입니다. 힘을 충분히 주어 발음하세요.',
        'ㅉ: This sound is produced by pressing the tongue firmly against the roof of the mouth and releasing it. Apply sufficient pressure for accurate pronunciation.'
    ),
    ('ㅍ', 'ㅂ/ㅎ'): (
        '받침 ㅂ과 초성 ㅎ이 만나면 자연스럽게 ㅍ으로 발음됩니다. ㅂ의 입술 닫힘을 유지하면서, ㅎ의 숨을 더해 부드럽게 ㅍ으로 발음하세요.',
        'When final ㅂ meets initial ㅎ, it naturally changes to ㅍ due to aspiration. Maintain the lip closure of ㅂ while adding the breathy airflow of ㅎ to produce a clear ㅍ sound.'
    ),
    ('ㅈ', 'ㅉ'): (
        'ㅈ은 입천장 가까이에서 가볍게 공기를 터뜨리면서 나는 소리입니다. 너무 강하게 터뜨려서 ㅉ처럼 되지 않도록 조절하세요.',
        'ㅈ: This sound is produced by lightly releasing air near the roof of the mouth. Avoid making it too strong, which may turn it into ㅉ.'
    ),
    ('ㄲ', 'ㄱ'): (
        'ㄲ은 혀뿌리에 힘을 주어 단단히 막았다가 터뜨리면서 나는 소리입니다. 힘을 약하게 주면 ㄱ처럼 들릴 수 있으니 조심하세요.',
        'ㄲ: This sound is made by pressing the back of the tongue firmly against the roof of the mouth and releasing it. If not pronounced strongly enough, it may sound like ㄱ, so be cautious.'
    ),
    ('ㄲ', 'ㅋ'): (
        'ㄲ은 세게 막았다가 터뜨리는 소리지만, ㅋ처럼 숨이 새어 나가면 안 됩니다. 공기를 너무 많이 내보내지 않도록 조절하세요.',
        'ㄲ vs. ㅋ: ㄲ is a strong stop sound, but it should not be aspirated like ㅋ. Avoid letting out too much air.'
    ),
    ('ㄸ', 'ㅌ'): (
        'ㄸ은 혀끝을 윗잇몸에 단단히 붙였다가 떼면서 나는 소리입니다. 숨이 새어 나가지 않도록 조심하세요.',
        'ㄸ: This sound is made by pressing the tongue tip against the upper gums and releasing it. Be careful not to let air escape during pronunciation.'
    ),
    ('ㅉ', 'ㅊ'): (
        'ㅉ은 입천장 가까이에서 단단히 막았다가 터뜨리면서 나는 소리입니다. ㅊ처럼 숨이 세게 나오지 않도록 조절하세요.',
        'ㅉ vs. ㅊ: ㅉ is made by strongly pressing and releasing the tongue near the roof of the mouth. Make sure not to add excessive airflow, which could turn it into ㅊ.'
    )
}

SUBSTITUTION_FEEDBACK = {
    ('ㄹ', 'ㄴ'): (
        'ㄹ은 혀끝을 윗잇몸에 가볍게 닿았다가 떼면서 나는 소리입니다. ㄴ처럼 입과 코 안에서 울리면서 나지 않도록 구별해서 발음하세요.',
        'ㄹ vs. ㄴ: ㄹ is made by lightly touching the tongue tip to the upper gums and releasing it. Avoid nasalizing it like ㄴ.'
    ),
    ('ㄴ', 'ㄹ'): (
        'ㄴ은 혀끝을 윗잇몸에 붙이고 입과 코 안에서 울려 나는 소리입니다. ㄹ처럼 가볍게 닿았다가 떼지 않도록 주의하세요.',
        'ㄴ vs. ㄹ: ㄴ is a nasal sound, resonating in both the mouth and nose. Avoid pronouncing it as a quick tap like ㄹ.'
    )
}

GROUP_SUBSTITUTIONS = {
    ('ㅙ', ('ㅗ', 'ㅐ')): (
        'ㅙ은 입술을 둥글게 모으면서 나는 소리입니다. ㅗ 하나로 줄이거나 ㅗ ㅐ처럼 길게 늘이지 않도록 조심하세요.',
        'ㅙ: This sound is pronounced with rounded lips. Avoid shortening it to ㅗ or pronouncing it as an extended ㅗㅐ.'
    )
}


INSERTION_FEEDBACK = {
    'ㅁ': (
        'ㅁ은 입술을 완전히 닫았다가 떼면서 나는 소리입니다. 원래 없던 ㅁ이 추가되지 않도록 입술 움직임을 조절하세요.',
        'ㅁ: This sound is made by completely closing the lips and releasing. Ensure no unintended ㅁ sound is added in speech.'
    ),
    'ㅔ': (
        'ㅔ는 입을 중간 정도 벌리고 내는 소리입니다. 필요 없는 소리가 끼어들지 않도록 주의하세요.',
        'ㅔ: This sound is produced by moderately opening the mouth. Avoid inserting unnecessary sounds.'
    ),
    'ㄴ': (
        'ㄴ은 혀끝을 윗잇몸에 대고 내는 비강을 울리면서 내는 소리입니다. 불필요한 ㄴ이 끼어들지 않도록 혀의 움직임을 신경 써 주세요.',
        'ㄴ (nasal sound): This sound is made by pressing the tongue against the upper gums and resonating in the nasal cavity. Be mindful not to insert extra ㄴ sounds.'
    ),
    'ㅊ': (
        'ㅊ은 입천장 가까이에서 공기를 세게 터뜨리면서 나는 소리입니다. 원래 없던 ㅊ이 추가되지 않도록 발음의 흐름을 부드럽게 이어 주세요.',
        'ㅊ (aspirated sound): This sound is made by forcefully releasing air near the roof of the mouth. Ensure no extra ㅊ sound appears in pronunciation.'
    )
}

DELETION_FEEDBACK = {
    'ㄷ': (
        'ㄷ은 혀끝을 윗잇몸에 붙였다가 떼면서 내는 소리입니다. 받침 ㄷ이 빠지지 않도록 혀끝을 확실히 붙였다가 떼어 주세요.',
        'ㄷ: This sound is made by pressing the tongue tip against the upper gums and releasing it. Ensure the final ㄷ sound does not get omitted.'
    )
}


def map_phonemes_to_words(correct_phoneme, correct_sentence):
    """
    정답 문장의 단어와 음소를 매핑하는 함수
    - 단어별로 음소를 정확히 배분
    """
    phonemes = correct_phoneme.lower().split()
    words = correct_sentence.lower().split()
    
    word_phoneme_map = {}
    phoneme_index = 0
    total_phonemes = len(phonemes)

    for word in words:
        word_phonemes = []

        # 단어당 음소 배정 (비율 기반 정규화)
        avg_phonemes_per_word = max(1, total_phonemes // len(words))
        while phoneme_index < len(phonemes) and len(word_phonemes) < avg_phonemes_per_word:
            word_phonemes.append(phonemes[phoneme_index])
            phoneme_index += 1

        word_phoneme_map[word] = word_phonemes

    return word_phoneme_map

def find_word_for_phoneme(word_phoneme_map, phoneme_index):
    """
    주어진 phoneme_index가 속하는 단어를 찾음
    """
    cumulative_index = 0
    closest_word = None

    for word, phonemes in word_phoneme_map.items():
        if cumulative_index <= phoneme_index < cumulative_index + len(phonemes):
            return word
        if phoneme_index < cumulative_index + len(phonemes):
            closest_word = word  # 가장 가까운 단어 저장
        cumulative_index += len(phonemes)

    return closest_word  if closest_word else "" # 매칭되는 단어가 없을 경우 가장 가까운 단어 반환


def analyze_phoneme_errors(correct_phoneme, predicted_phoneme, correct_sentence):
    from Levenshtein import editops

    correct_tokens = correct_phoneme.strip().split()
    predicted_tokens = predicted_phoneme.strip().split()
    word_phoneme_map = map_phonemes_to_words(correct_phoneme, correct_sentence)

    feedback_list = []
    ops = editops(correct_tokens, predicted_tokens)
    skip_indices = set()

    for op_index, (op, i, j) in enumerate(ops):
        if j in skip_indices:
            continue
        word = find_word_for_phoneme(word_phoneme_map, i)

        if op == 'replace':
            pair = (correct_tokens[i], predicted_tokens[j])
            if pair in VARIANTS_FEEDBACK:
                kr, en = VARIANTS_FEEDBACK[pair]
                feedback_list.append(f"- 발음 변이(Variants) 오류 발생: {pair}\n  ▸ [KR] {kr}\n  ▸ [EN] {en}")
            elif pair in SUBSTITUTION_FEEDBACK:
                kr, en = SUBSTITUTION_FEEDBACK[pair]
                feedback_list.append(f"- 음소 대체(Substitution) 오류 발생: {pair}\n  ▸ [KR] {kr}\n  ▸ [EN] {en}")
            else:
                # 복합 음소 체크 (1:2) 예: ㅙ → ㅗ ㅐ
                predicted_seq = predicted_tokens[j:j+2]
                for (target_ph, pred_group), (kr, en) in GROUP_SUBSTITUTIONS.items():
                    if correct_tokens[i] == target_ph and predicted_seq == list(pred_group):
                        feedback_list.append(
                            f"- 음소 분해(Substitution) 오류 발생 in '{word}': {target_ph} → {' '.join(pred_group)}\n"
                            f"  ▸ [KR] {kr}\n"
                            f"  ▸ [EN] {en}"
                        )
                        skip_indices.update({j, j+1})
                        break

        elif op == 'delete':
            phoneme = correct_tokens[i]
            if phoneme in DELETION_FEEDBACK:
                kr, en = DELETION_FEEDBACK[phoneme]
                feedback_list.append(f"- 음소 삭제(Deletion) 오류 발생: {phoneme}\n  ▸ [KR] {kr}\n  ▸ [EN] {en}")

        elif op == 'insert':
            inserted = predicted_tokens[j]
            next_inserted = predicted_tokens[j+1] if j + 1 < len(predicted_tokens) else None
            for (target_ph, pred_group), (kr, en) in GROUP_SUBSTITUTIONS.items():
                if [inserted, next_inserted] == list(pred_group):
                    word = find_word_for_phoneme(word_phoneme_map, i)
                    feedback_list.append(
                        f"- 음소 삽입(Substitution) 오류 발생 : {target_ph} → {' '.join(pred_group)}\n"
                        f"  ▸ [KR] {kr}\n"
                        f"  ▸ [EN] {en}"
                    )
                    skip_indices.update({j, j+1})
                    break

            if inserted in INSERTION_FEEDBACK:
                kr, en = INSERTION_FEEDBACK[inserted]
                feedback_list.append(f"- 음소 삽입(Insertion) 오류 발생 in '{word}': {inserted}\n  ▸ [KR] {kr}\n  ▸ [EN] {en}")

    if not feedback_list:
        feedback_list.append("문제있는 발음을 찾지 못하였습니다!")

    return feedback_list


"""

def analyze_phoneme_errors(correct_phoneme, predicted_phoneme, correct_sentence):

    # 정답 음소와 예측 음소를 비교하여 오류 유형을 분석하고 피드백을 제공합니다.
    # 오류 발생 시 해당 단어를 함께 제공.

    correct_tokens = correct_phoneme.strip().split()
    predicted_tokens = predicted_phoneme.strip().split()
    word_phoneme_map = map_phonemes_to_words(correct_phoneme, correct_sentence)

    feedback_list = []
    ops = editops(correct_tokens, predicted_tokens)

    for op, i, j in ops:
        word = find_word_for_phoneme(word_phoneme_map, i)

        if op == 'replace':
            pair = (correct_tokens[i], predicted_tokens[j])
            if pair in VARIANTS_FEEDBACK:
                feedback_kr, feedback_en = VARIANTS_FEEDBACK[pair]
                feedback_list.append(f"- 발음 변이(Variants) 오류 발생: {pair}\n  ▸ [KR] {feedback_kr}\n  ▸ [EN] {feedback_en}")
            elif pair in SUBSTITUTION_FEEDBACK:
                feedback_kr, feedback_en = SUBSTITUTION_FEEDBACK[pair]
                feedback_list.append(f"- 음소 대체(Substitution) 오류 발생: {pair}\n  ▸ [KR] {feedback_kr}\n  ▸ [EN] {feedback_en}")

        elif op == 'delete':
            phoneme = correct_tokens[i]
            if phoneme in DELETION_FEEDBACK:
                feedback_kr, feedback_en = DELETION_FEEDBACK[phoneme]
                feedback_list.append(f"- 음소 삭제(Deletion) 오류 발생: {phoneme}\n  ▸ [KR] {feedback_kr}\n  ▸ [EN] {feedback_en}")

        elif op == 'insert':
            inserted_phoneme = predicted_tokens[j]
            if inserted_phoneme in INSERTION_FEEDBACK:
                feedback_kr, feedback_en = INSERTION_FEEDBACK[inserted_phoneme]
                feedback_list.append(f"- 음소 삽입(Insertion) 오류 발생 in '{word}': {inserted_phoneme}\n  ▸ [KR] {feedback_kr}\n  ▸ [EN] {feedback_en}")


    for i in range(len(correct_tokens)):
        for (correct_phoneme, predicted_group), (feedback_kr, feedback_en) in GROUP_SUBSTITUTIONS.items():
            group_len = len(predicted_group)
            if correct_tokens[i] == correct_phoneme and predicted_tokens[i:i + group_len] == list(predicted_group):
                word = find_word_for_phoneme(word_phoneme_map, i)
                feedback_list.append(
                    f"- 음소 대체(Substitution) 오류 발생 in '{word}': {correct_phoneme} → {' '.join(predicted_group)}\n"
                    f"  ▸ [KR] {feedback_kr}\n"
                    f"  ▸ [EN] {feedback_en}"
                )

    if not feedback_list:
        feedback_list.append("문제있는 발음을 찾지 못하였습니다!")

    return feedback_list
"""