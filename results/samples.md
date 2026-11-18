# Sample completions

Model: 1,065,192-parameter GPT, val loss 3.3282 (perplexity 27.9) vs bigram 4.5304. 40 new tokens per completion, seed 0, prompt in **bold**.

## focused (T=0.5, top-k 20)

1. **The femur is** the cilian tubula, the calseale, which is the ap of the pelvis and is the flexorus and the axis.
2. **The heart is a hollow muscular organ**, and the heart, the blood, the heart, which is drawn from the veins. The walls of the pulmonary veins, and veins, which are the capillaries, the heart
3. **The function of the kidneys is** formed by the system. The TINERGLRARANES are the DRES is the naments of the linedge of
4. **Blood is composed of** the body, which is called the body. There is fracture of the stomach, and the system by the skin of the body, and the system. The nerves are the body,
5. **The lungs are** the lungs, in the lungs, and the heart, the venous blood, and the blood is not only ventricle, as the blood-vessels. The lungs are the blood,
6. **The skin consists of** the body, are called the liningugular vessel. The skull is the cerebellum, and is formed, but the nasal bones, two parts
7. **The spinal cord**, and the viscal cords, which are fracted by the brain, and the cerebrum, and the posterior tongue. The external ear is
8. **Digestion begins in the mouth, where** the blood is in the blood is not piece of the system. The brain is a varies by the skin of the brain, which, and other parts of the body, which are
9. **The pericardium**, and the inner surface of the posterior surface of the line, which forms the cerebrum, and is the fissure. The SALA
10. **Muscles are attached to** the cranial nerves. The PESTELRARANES are the brain, and several posteriorly. The SARA

## creative (T=1.0, top-p 0.9)

1. **The femur is** formed to four are surrounded by the enlarged of each side. The apange, when-brous male-so-cle, two ro
2. **The heart is a hollow muscular organ**, and caused from the blood at the heart, which carried masses all a little saliva, and seem, maint. The external corpuscles
3. **The function of the kidneys is** formed by a greater than its arranged inner vessel. In general organs are nearly by the skin, several action, and other articles of a part. 
4. **Blood is composed of** cartilaginous layers. It is there is larger and smallest, which form a layer by the middle surface is posterior. The concleule of
5. **The lungs are** formed to four are surrounded by the enlarged of each side. The apange, when-brous male. The concloped tub
6. **The skin consists of** the blood, are surrounded from the enlarged matter; only by the system, "By" OCr,", and other articles of correkish
7. **The spinal cord**, and causes from the different parts of the medium, and the masses, and the cerebrganic and is posterior. The external layers of the
8. **Digestion begins in the mouth, where** the blood causes cold as its applied to enfinity of heat, which all a persons by the skin of seem to maintain our concent, accum
9. **The pericardium**, and causes an organization of the enlarged matter; only by the system, "By" of several pioisonous apply," 
10. **Muscles are attached to** the cervix nerves of its arribes to the anterior and masses, and the celerals. The size is posterior ligaments of three layers. O

## define mode (glossary style `Term:`, stops at the end of the sentence)

- **pericardium** → A term applied to the lense of the cerebrum, which the ap of the back of the pelvis, and ischium.
- **aorta** → A term applied to the lense of the cerebrum, which the ap of the back of the pelvis, and ischium.
- **larynx** → A term applied to the lense of the cranial membrane of the stomach, and the cerebrum.
- **tachycardia** → A term applied to the lense of the cerebrum.
- **diaphragm** → A term applied to the lense of the cranial membrane of the stomach.

## Honest note on quality

- **What it learned:** the books' vocabulary and register — real terms in plausible places (ventricle, capillaries, pulmonary veins, cerebrum, ligaments), sentence rhythm, and the glossary's definition opener "A term applied to …".
- **What it did not learn:** facts. Completions are fluent-looking but usually wrong or meaningless, with invented words ("calseale", "concleule") and run-on lists.
- **Low temperature** repeats safe words ("the body … the system"); **high temperature** invents more words. 0.6–0.8 with top-k/top-p is the best trade-off.
- **define** produces the same generic glossary-style sentence for every term: 428 definitions are far too few for a 1M-parameter model to tie terms to meanings.
- Odd ALL-CAPS fragments come from capitalised headwords in Cutter's text.
- Same seed = same random draws, so different prompts can drift into the same continuation.

Use it to practise spotting and correcting terminology, never as a source of facts.
