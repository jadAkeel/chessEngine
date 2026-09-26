# دراسة عيوب وحدود MCP / OpenCode Orchestrator

تاريخ الفحص: **2026-09-05**. هذه الدراسة تحفظ حالة النظام وقت الفحص كمرجع تاريخي. طُبّقت بعدها إصلاحات ونُشرت نسخة محمية جديدة بعد اجتياز بوابات الاختبار؛ التفاصيل والقيود المتبقية موجودة في [`AUDIT_FIXES_2026-09-05.md`](C:/Users/10User/codex-opencode-mcp/docs/AUDIT_FIXES_2026-09-05.md).

**تحديث 2026-09-06:** بعد موافقة المستخدم الصريحة على تخفيف عزل `--pure`، فُعّل `google/antigravity-gemini-3.8-flash` مع `high` كإعداد افتراضي للـagents غير المعزولة عبر MCP bridge. تم إصلاح عيوب الأدلة، health compatibility، التدقيق، السباقات، والنشر المذكورة في هذه الدراسة، واجتاز الإصدار الجديد اختبارات المصدر والإصدار وlive Gemini smoke. بقيت حدود معروفة: OpenCode لا يبث هوية runtime موثوقة في كل JSON stream، وOAuth runtime قابل للكتابة لذلك لا يساوي ضمان full immutable release. التفاصيل التنفيذية والـrollback hashes في ملف الإصلاح المرتبط أعلاه.

الاستنتاج الأساسي: النظام يملك ضوابط جيدة لتحديد نطاق الكتابة والتحقق من النتائج، لكن تجربة استخدامه تتأثر بثلاث مجموعات مختلفة من المشكلات: عيوب فعلية في معالجة الأدلة وإخفاء الأسرار، قيود مقصودة لا تتوافق مع طلب Gemini عبر plugin، وفجوة بين نسخة التطوير والنسخة المشغّلة. معالجة هذه المجموعات بالطريقة نفسها ستؤدي إلى إصلاحات غير لازمة أو إضعاف ضوابط نافعة.

النتائج الأهم المثبتة حديثاً: فوات صيغ GitHub من الإخفاء، الاحتفاظ بأول هوية موديل فقط، قبول preflight لمسار يرفضه التنفيذ، وفشل تشغيل حقيقي في إنتاج جواب نهائي رغم `exitCode=0`. لم نرصد تسريب سر حقيقي أو تبديل موديل حقيقي أو فقدان ملفات؛ اختبارات الأسرار والأحداث استخدمت بيانات اصطناعية فقط.

## 1. نطاق الدراسة ومنهج الإثبات

المقصود بـ«MCP orchestrator» هنا ليس بروتوكول MCP بحد ذاته، بل الـbridge المحلي `codex-opencode-mcp` مع ملفات agents، وOpenCode CLI، وإعدادات التشغيل. هذا الفصل ضروري لتحديد مكان الإصلاح.

| الطبقة | مسؤوليتها | مثال على عيب أو قيد فيها |
| --- | --- | --- |
| Codex، منسّق المهمة | اختيار الأدوات وتقسيم العمل ومراجعة المخرجات | اختيار agent لا يطابق حدود القراءة |
| MCP bridge | التحقق، التشغيل، queue، locks، worktrees، الدمج | preflight ناقص أو تقرير موديل غير مكتمل |
| `mcp-orchestrator` | قراءة وتخطيط داخل الصلاحيات الممنوحة | محاولة أمر shell خارج القائمة أو غياب جواب نهائي |
| OpenCode CLI | تنفيذ الأدوات وإصدار الأحداث والتواصل مع المزوّد | اختلاف شكل الأحداث أو سلوك رفض الصلاحيات |
| إعدادات المزوّد والموديل | تحديد النقل والمصادقة والموديل | Gemini عبر plugin مع release يفرض pure mode |

استخدمت الدراسة قراءة الكود الفعلي، قراءة الحقول غير السرية من إعدادات التشغيل، أدوات MCP الحية، اختبارات المصدر الموجودة، وprobes محلية تستخرج الدوال الفعلية من المصدر بدون تشغيل خادم MCP. استعنت بتوثيق OpenCode وGitHub الرسمي في النقاط المتعلقة بسلوكهما الخارجي.

تصنيفات الأدلة:

- **مثبت حيّاً:** ظهر في استدعاء MCP أثناء هذه الدراسة.
- **مثبت باختبار اصطناعي:** أعيد إنتاجه على الدالة الفعلية بمدخلات وهمية؛ لا يثبت حدوثه مع مزوّد حقيقي.
- **مثبت من الكود:** مسار التنفيذ واضح، لكن السيناريو الكامل لم يُشغّل.
- **فرضية تحتاج قياساً:** تفسير محتمل أو أثر متوقع، وليس حادثة مثبتة.

الأولويات: **P1** قبل الاعتماد على الميزة المتأثرة؛ **P2** تحسين موثوقية وتشغيل؛ **P3** تحسين صيانة أو راحة استخدام. لم يُثبت عيب P0 في هذا الفحص.

## 2. النسخ والإعدادات التي فُحصت

| المرجع | القيمة |
| --- | --- |
| `R`، النسخة المشغّلة | `C:/Users/10User/codex-opencode-mcp-releases/server-3e828b6c-20260813-lock-origin` |
| SHA-256 لملف `R/server.js` | `3e828b6cbf52bb961ebb4b8775e0ee742a489cf57fc50fabebcd14cca1980679` |
| SHA-256 المثبت للـrelease manifest في إعداد MCP | `2925caca1525400144a8f7a3931917061b9dc4e4b576c47cfbe931c88878899d` |
| `D`، نسخة التطوير | `C:/Users/10User/codex-opencode-mcp` |
| آخر commit في `D` وقت الفحص | `16bbf7b`؛ توجد تعديلات غير committed، لذلك الـcommit وحده لا يعرّف النسخة المفحوصة |
| SHA-256 لملف `D/server.js` | `9b6b3fce153c6d145aa3cf400c3b59f2f33969e01ea9d82501170a12ac7d17ce` |
| OpenCode / Git / Node | `1.17.13` / `2.39.1.windows.1` / `v24.11.1` |
| النظام | Windows / PowerShell |
| إعداد orchestrator الفعلي | `openai/gpt-5.6-terra`، `high`، وضع `planning-only` |
| الإعداد الشخصي لـOpenCode | default هو `google/antigravity-gemini-3.8-flash`؛ `general/plan/explore` على `high` |
| plugin الشخصي | `@cortexkit/opencode-antigravity-auth@2.2.1` |
| إعداد plugins في MCP | `CODEX_OPENCODE_ALLOW_EXTERNAL_PLUGINS=false`، وتشغيل `--pure` |
| التشغيل والتوازي | `worktreeMode=write`، `queueMode=sqlite`، حد المزوّد `2` |
| locks | قراءة `off`؛ كتابة منفردة `simple`؛ كتابة متوازية `strict` |
| contractor | profile موجود، لكن capability غير مهيّأة في التشغيل الحالي |

إعداد MCP يشير فعلياً إلى `R/server.js` ويضبط `XDG_CONFIG_HOME` على الـrelease. لذلك تغيير default في الإعداد الشخصي لا يغيّر موديل الـbridge. المرجع: [إعداد MCP][config]، [إعداد release][release-config]، [profile التخطيط][orchestrator-profile].

اسم `google/antigravity-gemini-3.8-flash` أعلاه هو معرّف في الإعداد الشخصي. لم تتحقق هذه الدراسة من مطابقته لمعرّف Google API رسمي، أو من صحة المصادقة عليه. يجب الفصل بين نموذج المساعد البرمجي وبين Gemini الخصم في تطبيق الشطرنج؛ لكل منهما مسار إعداد وتشغيل مختلف.

## 3. تصحيحات للتقييم السابق

| الانطباع السابق | ما أثبته الفحص الأعمق |
| --- | --- |
| «لا يوجد plugin allowlist» | موجود مع تثبيت الإصدار والملفات والمصادر بالـhash. لكنه **ممنوع مع immutable release pinning** الحالي. |
| «كل قراءة تتطلب Git» | المسار العادي يتطلبه؛ يوجد `sanitizedWorkspace` موثّق بالـmanifest يعمل بدون Git. |
| «عدم اكتشاف agent محلي خلل discovery» | تعطيل project config مقصود في `buildOpenCodeEnv` لحماية حدود التشغيل. القصور في عرض هذا الفرق وتوفير تخصيص موثوق. |
| «رفض general/plan عطل» | رفض `general` كقارئ صحيح لأنه قابل للكتابة؛ `plan` رُفض بسبب الصلاحيات الفعلية، وليس بسبب اسمه. |
| «orchestrator لا ينسق agents» | profile التخطيط يمنع nested tasks عمداً؛ يوجد contractor مختلف، لكنه غير مفعّل هنا. الـbridge نفسه يدعم تشغيل jobs متوازية. |
| «دائماً بطيء ويحتاج 3.4 دقائق» | رقم سابق من ملخص المحادثة، وليس benchmark. التجربة الجديدة انتهت خلال `30,749 ms` لكن بدون جواب نهائي. لا تصح مقارنة الزمنين كأداء لنفس المهمة. |

يجب الحفاظ على رفض الصلاحيات غير الآمنة، وعدم حل مشكلة الاستخدام بتحويل `deny` إلى `allow` على نحو عام.

## 4. سجل النتائج وأولوياتها

| ID | النتيجة | الأولوية | الدليل | الوضع في نسخة التطوير |
| --- | --- | --- | --- | --- |
| F01 | صيغ GitHub غير الموسومة تفلت من redaction | P1 | probe على النسختين | ما زالت موجودة |
| F02 | إثبات الموديل ناقص ويتجاهل الهوية اللاحقة | P1 | كود + probe + غياب الدليل في التشغيل الحي | ما زالت موجودة |
| F03 | parser يحتفظ بآخر text فقط ويتسامح مع سطر غير صالح | P2 | probe؛ الأثر الحي غير مثبت | ما زالت موجودة |
| F04 | preflight لا يثبت جاهزية Git/HEAD للتنفيذ | P1 | قبول حي ثم رفض حي + كود | يحتاج إعادة اختبار endpoint في `D` |
| F05 | تشغيل فعلي انتهى بلا جواب نهائي | P2 | تجربة حية | سبب CLI الجذري غير محسوم |
| F06 | طلب موديل/variant لكل job غير ممثل في واجهة MCP | P1 لمتطلب Gemini | schemas + إعدادات فعلية | لم يُثبت دعم جديد له |
| F07 | تعارض plugin OAuth مع immutable release | P1 لمتطلب Gemini الحالي | كود + probe | التعارض موجود |
| F08 | النسخة المنشورة متأخرة عن تحسينات التطوير | P1 للنشر | مسار التشغيل + hashes + مقارنة محددة | تحسينات غير منشورة |
| F09 | التشغيل المباشر لا يدخل سجل queue التشخيصي | P2 | تجربة حية + كود | لا نفترض حله بدون اختبار |
| F10 | القراءة المتوازية من checkout متغير تضعف نسبة التغييرات | P2 | كود؛ لم نعد إنتاج سباق حي | يلزم اختبار مستقل |
| F11 | احتفاظ غير محدود افتراضياً بالـqueue في `R` | P2 | كود | retention وحدود سعة مضافة في `D` |
| F12 | احتواء العملية في `R` أضعف من supervisor الحديث | P1 قبل توسيع الكتابة غير المراقبة | كود مقارن؛ لا حادثة تسرب عملية مثبتة | تحسينات موجودة؛ Windows ما زال best-effort |

### F01 — خلل محدد في إخفاء صيغ GitHub

**الملاحظة:** `redactSensitiveText` يضع `ghp` و`gho` و`github_pat` داخل مجموعة متبوعة بشرطة `-`، بينما صيغ GitHub المعنية تستخدم `_`. السلسلة الاصطناعية غير الموسومة بقيت كما هي على النسختين. أما وضعها بعد `api_key=` فتم إخفاؤه بسبب قاعدة أخرى؛ لذلك العيب مشروط بطريقة ظهور القيمة، وليس فشل كل الإخفاء. تؤكد [وثائق GitHub الرسمية][github-token-doc] بادئتي `ghp_` و`github_pat_`.

**الأثر:** إذا ظهر token بصيغته الخام في جواب أو نص يُمرَّر إلى هذه الدالة، يمكن أن يبقى في المخرج المنقّح. لم نستخدم أي token حقيقي، ولم نثبت أن سراً حقيقياً وصل إلى log أو queue.

**الإصلاح المقترح:** فصل عائلات البوادئ إلى أنماط صحيحة، وإضافة canaries آمنة لكل صيغة ومسار إخراج. لا تعتمد السرية على regex وحده؛ تقليل الملفات والبيانات التي يراها الوكيل يبقى ضرورياً.

**معيار القبول:** اختبارات خام وموسوم وداخل URL/JSON/نص نهائي؛ عدم ظهور canary في المخرجات المسموح بحفظها، مع حالات سالبة تمنع الإخفاء المفرط للنص الطبيعي. الملفات: [R/server.js:633][redaction-r] و[D/server.js:784][redaction-d].

### F02 — تثبيت الموديل لا يساوي إثبات جميع رسائل التشغيل

**الملاحظة:** الـCLI يحصل على `--model` و`--variant` صريحين، وهذه نقطة قوة. لكن `inspectOpenCodeEventStream` يستخدم `runtimeModelEvidence ||= ...`، فيحتفظ بأول هوية فقط. أدخل probe حدثين موثوقين اصطناعياً: `model-a` ثم `model-b`؛ بقيت النتيجة `model-a`. لا يثبت هذا أن تبديل موديل حصل فعلياً.

كذلك، عندما لا تصدر هوية، يعرض النظام `not_runtime_emitted` ولا يرفض لهذا السبب وحده. رفض mismatch في `runOpenCode` مشروط بوجود هوية. التجربة الحية الجديدة لم تصدر هوية، لكن سبب فشلها النهائي كان غياب الجواب، وليس هوية الموديل. لا يوجد في الأدلة المفحوصة إثبات runtime للـvariant نفسه.

**الأثر:** عبارة «Silent model fallback: disabled» تصف سياسة الطلب والتثبيت؛ لا تثبت أن كل استجابة صدرت من الموديل المطلوب. لا يجوز تحويلها إلى ادعاء «Gemini high اشتغل فعلياً».

**الإصلاح المقترح:** تجميع أدلة الهوية على مستوى session/message، والتحقق من جميع رسائل المساعد المعنية مع تمييز parent عن nested agents. إضافة سياسة صريحة مثل `requireRuntimeModelEvidence`، وإرجاع requested/configured/observed/confidence كحقول مستقلة. عند غياب الدليل يكون التشغيل «نجاح بدون إثبات هوية» أو رفضاً إذا اشترط الطلب ذلك.

**معيار القبول:** mismatch في أول أو آخر رسالة يُكتشف؛ غياب الهوية لا يُعرض كإثبات؛ معلومات الأدوات أو كلام النموذج عن نفسه لا تُعامل كدليل. المرجع: [parser][events-r] و[تحقق نتيجة التشغيل][run-evidence-r].

### F03 — حدود سلامة واكتمال stream

**الملاحظة:** كل حدث text مكتمل يستبدل `finalText`. في probe لجزأين يحملان `messageID` نفسه ضاع الجزء الأول. وفي probe آخر، سطر JSON غير صالح ثم text نهائي أعطيا `invalidLines=1` دون خطأ من classifier. لم نثبت أن OpenCode المستخدم يصدر فعلياً النص النهائي مجزأ بهذه الصورة؛ هذه حدود توافق مثبتة عند مستوى parser.

**الأثر:** احتمال جواب ناقص أو نجاح مع stream غير مفهوم جزئياً عند تغيّر إصدار CLI أو شكل الأحداث. وجود نص أخير لا يكفي دائماً لإثبات أن تفسير السجل كان كاملاً.

**الإصلاح المقترح:** تجميع أجزاء الرسالة النهائية بحسب IDs ودورة حياتها، مع فصل commentary عن final. تصنيف الأسطر غير المفهومة إلى diagnostics مسموحة وأحداث تالفة، وإظهار `streamIntegrity` بدلاً من إسقاطها بصمت.

**معيار القبول:** fixtures مأخوذة من CLI الحقيقي ومُنقّحة، تغطي multipart، أداة بعد نص، نص فارغ، سطر مقطوع، حدث جديد، وأحداث متداخلة. لا يكفي وصل جميع النصوص لأنه قد يخلط التحديثات المؤقتة بالجواب النهائي. المرجع: [R/server.js:2898][events-r] و[R/server.js:3853][classifier-r].

### F04 — preflight ينجح فيما يرفض التنفيذ الشروط نفسها

**إعادة الإنتاج الحية:** إرسال read-only job للـ`mcp-orchestrator` إلى مجلد `R` غير المهيأ كمستودع Git أعطى `Delegation plan accepted`. إرسال الـjob نفسه إلى `run_opencode_agent` أعطى `git_state_required` خلال `75 ms`، قبل تشغيل المزوّد.

**السبب:** endpoint التخطيط يتحقق من العقد والـrouting والصلاحيات، لكنه لا يمر على كل شروط التنفيذ؛ `executeOpenCodeJob` يفحص Git لاحقاً. فحص root نفسه لا يثبت وجود commit، ثم `captureGitHead` يتطلب `HEAD` ويستخدم تصنيفاً اسمه `integration_target_state_failed` حتى في سياق قراءة. حالة repo بلا commit مثبتة من المسار البرمجي، ولم نكرر إنشاء repo لهذا السيناريو في هذه الدراسة.

**الإصلاح المقترح:** تقرير preflight متعدد النتائج: policy valid، workspace ready، provider configured، وما لم يُختبر. إضافة فحص root وHEAD وdirty-checkpoint في مرحلة مناسبة. عرض مسار `sanitizedWorkspace` كخيار جاهز للقراءة بدون Git؛ ليس ضرورياً اختراع هذا المسار من جديد.

**معيار القبول:** جدول اختبارات: non-Git، repo بلا commit، clean repo، dirty repo، sanitized valid/invalid. لا تعرض «جاهز للتنفيذ» إذا لم تُفحص شروطه. لا تنشئ Git أو commits تلقائياً لمجرد اجتياز gate. المرجع: [preflight][preflight-r]، [Git root][git-root-r]، [HEAD][head-r]، [execution][execute-r].

### F05 — exit code ناجح بدون جواب مفيد

**التجربة:** مراجعة مستقلة read-only لنسخة التطوير، عبر الـbridge المنشور، باستخدام `mcp-orchestrator`. النتيجة: `exitCode=0`، `agent_empty_final_response`، `30,749 ms`، لا تغييرات ملفات مكتشفة، ولا timeout أو provider error مُكتشف. السجل أظهر أدوات قراءة مكتملة، ثم رفض `git log --oneline --decorate -10` لأنه خارج قائمة الأوامر الحرفية المسموحة.

**ما نعرفه:** الـbridge اكتشف غياب الجواب ورفض النتيجة؛ هذه حماية صحيحة. **ما لا نعرفه:** هل رفض shell أنهى CLI، أم أن هناك سبباً آخر في دورة الأحداث؟ التقرير الحالي لا يحسم السبب. لا يجوز اختزالها إلى «Gemini تعطل»؛ النموذج المهيأ هنا Terra والدليل runtime غير متاح.

**الإصلاح المقترح:** تسجيل سبب نهاية session وتسلسل terminal events بصورة محدودة ومنقّحة، وإظهار `permission_denied` مع الأمر وبديله المسموح. مطالبة agent بإكمال تقرير جزئي عند منع أداة. احتفظ بسياسة عدم إعادة التشغيل تلقائياً بعد استخدام الأدوات إلا بعد إثبات أمان الإعادة.

**معيار القبول:** fixture لرفض أداة يتبعه جواب، وآخر بدون جواب؛ كلاهما يعرض السبب الصحيح. مقارنة أمر allowlisted حرفي مع نفس الأمر بوسيط إضافي. عدم اعتبار exit 0 وحده نجاحاً. الدليل الكامل المنقّح محفوظ في [live-observations.json](evidence/live-observations.json).

### F06 — تفضيل Gemini الشخصي لا ينتقل إلى jobs

**الملاحظة:** schema أدوات التشغيل المفحوصة لا يتيح اختيار `model` و`variant` لكل job. يتم استخراج القيم من profile الموثق وتمريرها إلى CLI. إعداد release أيضاً يحدد موديلات OpenAI، و`XDG_CONFIG_HOME` يعزلها عن الإعداد الشخصي. النتيجة هنا حتمية من الإعدادات؛ ليست دليلاً على fallback خفي.

**الإصلاح المقترح:** إضافة model policy من المشغّل، تسمح بتحديد المزوّد والموديل والـvariant دون منح agent صلاحيات أوسع. عرض مصدر كل قيمة: user request، managed default، provider capability. اجعل تغيير default محصوراً في fleet أو المشروع المطلوب، وليس إعداداً عاماً ضمنياً.

**معيار القبول:** الطلب الصريح لـGemini يُنفّذ به بعد تحقق availability، أو يُرفض قبل إنفاق tokens مع تفسير التعارض. لا يُشغّل Terra بديلاً صامتاً. اختبار وراثة variant ودعم المزوّد له. ملفات البداية: [schema التشغيل][run-tool-r] و[بناء CLI args][cli-args-r]. يدعم OpenCode نفسه flags للموديل والـvariant؛ القيد هنا في واجهة الـbridge. [توثيق CLI][opencode-cli]

### F07 — plugin موجود في التصميم لكنه غير متوافق مع release الحالي

**الملاحظة:** `verifyExternalPluginPolicy` يوفّر allowlist صارماً، لكن `immutableReleasePluginModeError` يرفض الجمع بين release-manifest pinning وexternal plugins. السبب الموثق: plugin OAuth المعني يخزن credentials تحت `XDG_CONFIG_HOME` نفسه المستخدم للإعدادات الثابتة. الـprobe أكد هذا الرفض في `R` و`D`.

**الأثر:** مجرد ضبط `CODEX_OPENCODE_ALLOW_EXTERNAL_PLUGINS=true` ليس حلاً؛ سيفشل startup في هذا الوضع. تعطيل hash pinning لتشغيل Gemini سيغير ضمانات التشغيل ويحتاج قرار تصميم منفصلاً.

**مساران للإصلاح لاحقاً:** تقييم Google API بمصادقة أصلية متوافقة مع pure mode، بعد التحقق من معرّف الموديل والبيئة؛ أو فصل إعدادات plugin الثابتة عن مخزن credentials المتغير إذا كان plugin يدعم ذلك، مع اختبارات refresh وrestart. توافر المسارين لم يُختبر هنا.

**معيار القبول:** refresh لا يغيّر ملفات الـrelease، تغييرات plugin غير المراجعة تُرفض، وتبقى أسراره خارج manifest والتقارير. إثبات provider/model الفعلي شرط مستقل. المرجع: [plugin verifier][plugins-r] و[منع وضع release][immutable-r].

### F08 — الإصلاح الموجود في checkout قد لا يصل إلى التشغيل

**الملاحظة:** `R` و`D` مختلفان فعلياً في الـhash والحجم والملفات. `D` يضيف `bin/process-supervisor.js` و`bin/mcp-robustness.js` و`bin/state-audit.js` وحدود احتفاظ وسعة. ملف `server.js` يحتوي `18,388` سطراً في `R` مقابل `23,752` في `D`. الأرقام تشمل الاختبارات المضمّنة، ولا تمثل حجم منطق الإنتاج وحده.

**الأثر:** تشغيل اختبارات `D` أو إصلاحه لا يثبت سلامة `R`. وفي المقابل، الترقية وحدها لا تصلح F01/F02/F03، لأن probes أعادت إنتاجها في `D` أيضاً.

**الإصلاح المقترح:** إظهار release ID وserver hash وmanifest hash وNode/OpenCode version في health ونتائج jobs، ثم مسار نشر مراجَع إلى snapshot جديد مع fresh-process health وخطة رجوع. يوجد بالفعل build-release/fresh-healthcheck؛ يجب استخدامهما وتطوير ما ينقصهما.

**معيار القبول:** بعد نشر مستقبلي، يؤكد client جديد أن العملية الجديدة تعمل بالـhash المستهدف؛ لا تكتفِ بقراءة ملف config. لا تُنشر تعديلات checkout غير مراجعة تلقائياً. المرجع: [إعداد MCP][config] و[حزمة التطوير][package-d].

### F09 — diagnostics لا يعرض تاريخ التشغيل المباشر

**الملاحظة:** بعد فشل المراجعة في F05، أعاد `diagnose_opencode_bridge` للمستودع نفسه `jobs=0` و`failedJobs=0`. في الكود، `run_opencode_agent` يستدعي التنفيذ مباشرة ويرجع الاستجابة، بينما queue لها سجل منفصل. كذلك تستخدم عمليات OpenCode التابعة `OPENCODE_DB=:memory:`؛ لا يوجد session DB دائم لهذا المسار بالتصميم.

**الأثر:** لا يعني «لا توجد failed jobs» أن كل المحاولات السابقة نجحت. يصعب استرجاع تشغيل مباشر أو ربطه بالأداء والأخطاء بعد انتهاء جلسة العميل. هذا نقص تغطية للتشخيص، وليس إثبات فقدان سجل queue موجود.

**الإصلاح المقترح:** سجل موحد صغير لكل run، بما فيه direct/parallel/preflight failures، بمعرّف correlation ونوع التنفيذ. أو استخدام enqueue للمهمات الطويلة، مع توضيح أن واجهة direct لا تقدم نفس history/cancellation. احفظ ملخصات منقّحة وحدود retention، وليس prompts الخام بلا حاجة.

**معيار القبول:** job مباشر فاشل يظهر في diagnostics بعد reconnect، مع errorType وهوية النسخة ومدة التشغيل؛ queue cancellation لا يُعرض كأنه يلغي أي عملية خارج queue. المرجع: [direct handler][run-tool-r] و[بيئة OpenCode][env-r].

### F10 — القراءة من checkout متغير تؤثر على التوازي

**الملاحظة:** في الإعداد الحالي، القراء يعملون بدون read lock وفي checkout الأصلي، بينما الكتابة المعزولة تستخدم worktrees. تغير الملفات يُحسب بمقارنة snapshots قبل وبعد. إذا عدّل المستخدم أو process آخر الملف أثناء القراءة، لا تستطيع مقارنة المسارات نسبة التغيير إلى الوكيل وحده.

الكود يعترف بذلك ويحتفظ بالتغييرات بدلاً من rollback آلي للتعديلات غير المنسوبة؛ هذه نقطة جيدة. لكنه قد يرفض قراءة بريئة بسبب تغيير خارجي، أو يعطي تحليلاً مبنياً على حالات متغيرة. لم نُجرِ تجربة سباق على ملفات المستخدم.

**الإصلاح المقترح:** لقارئ يحتاج نتيجة قابلة للإعادة، جهّز snapshot قراءة ثابتاً أو sanitized manifest. اذكر revision أو manifest في النتيجة. أبقِ حالات التغيير الخارجي منفصلة عن «الوكيل عدّل ملفاً».

**معيار القبول:** قارئ مع تعديل خارجي متزامن لا يمحو تعديل المستخدم، ويبلغ `external_state_changed` أو ما يعادله؛ اختبار قارئين مع writer معزول يثبت overlap بدون خلط الملكية. المرجع: [المقارنة والاحتفاظ بالتغييرات][snapshots-r].

### F11 — تضخم السجلات والعمل المحتفظ به

**الملاحظة:** default `queueRetentionDays=0` في `R` يعطل pruning، و`worktreeCleanup=never` في التشغيل الفعلي يحتفظ بالمخرجات حتى تُدمج وتُنظف صراحة. هذا مفيد للتحقيق والاسترجاع لكنه يسمح بالتراكم. لم نثبت نفاد مساحة أو تضخم queue للمشروع الحالي؛ diagnostics كانت فارغة.

**الفرق في `D`:** توجد defaults للاحتفاظ بالـqueue لمدة 30 يوماً، audit لمدة 90 يوماً، وحدود لقاعدة الحالة والعمل المحتفظ به. وجود الإعدادات لا يغني عن اختبار سلوكها بعد النشر.

**الإصلاح المقترح:** مراجعة التحسينات الحالية ونشرها بعد الاختبار، مع قياس مساحة الحالة وتنبيه قابل للتصرف. حافظ على worktrees الفاشلة التي تحتوي عملاً مطلوباً، ولا تربط تنظيفها بمجرد انقضاء وقت قصير.

**معيار القبول:** حذف أو أرشفة terminal records المنتهية فقط، استمرار تحديث cancellation/terminal عند بلوغ السعة، ورفض إنشاء عمل جديد برسالة واضحة. لا حذف لسجلات active أو لعمل غير مدموج. المرجع: [احتفاظ R][retention-r] و[إعدادات D][retention-d].

### F12 — احتواء subprocess عند الفشل ليس ضماناً موحداً

**الملاحظة من `R`:** التنفيذ ينشئ child مباشرة؛ في Windows يستدعي `taskkill /PID /T /F` ثم يكمل عند close أو بعد مهلة. لا يحمل هذا المسار supervisor مستقل حديثاً يراقب فقدان bridge authority. لا نستنتج من ذلك أن عملية يتيمة حدثت في تجربتنا.

**التحسين في `D`:** supervisor، إثبات هوية قبل launch، watchdog، وحجز الموارد عند عدم تأكيد إنهاء الشجرة. ومع ذلك تصف نسخة Windows ضمانها بأنه `windows_taskkill_best_effort`. لذلك حتى وجود supervisor ليس دليلاً على عزل OS كامل.

**الإصلاح المقترح:** مراجعة واختبار التحسين الموجود قبل نشره؛ تقييم Windows Job Objects أو containment مكافئ إذا كان ضمان إنهاء جميع الأبناء مطلباً. فصل `cancel_requested` عن `tree_termination_confirmed`، وعدم إعادة الموارد للتوزيع عند بقاء عملية غير مؤكدة.

**معيار القبول:** fixtures مع child وgrandchild، موت مفاجئ للـbridge، إلغاء أثناء launch، وفشل وسيلة القتل. لا PID reuse يقتل عملية غير مرتبطة. يجب أن يذكر التقرير مستوى الضمان حسب OS. المرجع: [تنفيذ R][spawn-r] و[supervisor في D][supervisor-d].

## 5. أوجه قصور إضافية لا نخلطها بعيوب مثبتة

| المجال | التقييم | العمل المقترح |
| --- | --- | --- |
| `.env.example` | `**/.env.*` ضمن forbidden edits، لذلك المثال يقع ضمن المنع العام. هذا تشدد متوقع من السياسة، وليس عطل parser. | استثناء exact-path موثوق لملفات أمثلة مدققة، مع بقاء أسرار `.env` ممنوعة. لا استثناء عام بالاسم وحده. |
| صلاحيات shell | أوامر حرفية مثل `git log --oneline --decorate` لا تشمل تلقائياً نسخة بوسيط `-10`؛ ظهرت هذه الحالة فعلياً. | عرض الأوامر المقبولة للوكيل، أو سياسة parsing لأوامر قراءة محدودة بدل wildcard عام. |
| نطاق القراءة والسرية | `forbiddenEdits` يثبت منع الكتابة، ولا يثبت وحده أن كل قراءة سرية ممنوعة. تختلف الضمانات حسب profile وsanitized mode. | اختبارات read-canary بملفات وهمية؛ استخدام sanitized workspace عند الحاجة لحدود ملفات دقيقة. لا دليل هنا على قراءة سر فعلي. |
| health | كلمة healthy تختبر الأدوات والـprofiles، وليست استدعاءً ناجحاً للموديل أو شهادة جودة استجابته. | فصل installation health عن workspace readiness وprovider readiness، مع تكلفة أي smoke call ظاهرة. |
| السرعة | لا تتوافر عينة متجانسة لاستخراج p50/p95. يوجد عدة استدعاءات metadata وskill checks وsnapshots حول run. | قياس أزمنة discovery/attestation/snapshot/provider-wait/generation/validation منفصلة قبل تحسينها. لا حذف فحص pre-spawn لمجرد تسريع التشغيل. |
| «التوازي» | حد provider الحالي 2، حتى لو كانت حدود jobs أكبر. وplanning-only لا يسمح بتفويض داخلي. | قياس تداخل child execution intervals الفعلي، مع عرض سبب انتظار كل job. الدوال المساعدة لإثبات overlap موجودة أصلاً. |
| validation | `git diff --check` لا يثبت صحة ميزة Python/React. تشغيل test command يخضع لسياسة executable موثوقة. | recipe تحقق لكل نوع مشروع؛ لا تمرير shell arbitrary لتجاوز allowlist. |
| maintainability | `server.js` يجمع runtime وschemas وqueue ودمج Git واختبارات كثيرة في ملف كبير. | فصل تدريجي بعد تثبيت اختبارات characterization؛ لا إعادة كتابة شاملة ضمن إصلاح عيب صغير. |
| cache/OS isolation | hashes وworktrees تمنع أنواعاً من drift لكنها ليست sandbox ضد process يملك صلاحيات المستخدم نفسها. | توثيق threat model؛ تحسين العزل عند طلبه، دون ادعاء sandbox غير موجود. |

## 6. ما يعمل جيداً ويجب الحفاظ عليه

- التحقق من effective permissions وإعادة attestation قبل spawn، ورفض agent قابل للكتابة تحت عقد قراءة.
- تمرير الموديل والـvariant صراحة إلى CLI، واكتشاف native fallback؛ يلزمهما استكمال دليل runtime كما في F02.
- فصل كتابة agents في worktrees، وتحديد ownership وshared/forbidden/serial paths.
- receipt مراجعة مرتبط بهوية المصدر والهدف والـpatch قبل الدمج، مع فحوص تغيّر الحالة.
- الاحتفاظ بالتغييرات غير المنسوبة بدلاً من محو عمل المستخدم تلقائياً.
- retries محدودة ومشروطة؛ عدم إعادة writer أو قراءة استخدمت الأدوات بصورة عمياء.
- sanitized manifest يرفض ملفات غير متوقعة وروابط ومسارات غير آمنة، ويفتح مسار قراءة بدون Git.
- الاختبار الحي في F05 لم يتحول إلى نجاح مزيف رغم exit 0.

## 7. خطة الإصلاح المقترحة

هذه قائمة عمل لاحقة؛ المربعات غير المؤشرة تعني أن الإصلاح لم يُنفّذ في هذه الدراسة.

| المرحلة | العمل | التبعية | مخرج المراجعة المطلوب |
| --- | --- | --- | --- |
| A | تثبيت baseline للنسخة المشغلة ونسخة التطوير | لا شيء | hashes ونسخ الأدوات ونتائج فحص معروفة؛ بدون تفعيل checkout غير مراجع |
| B | F01: تصحيح redaction واختبارات canary | A | patch صغير واختبارات مسارات الإخراج |
| B | F02/F03: إصلاح أدلة الموديل وتفسير الأحداث | A؛ يمكن بحثه بالتوازي مع F01 | fixtures حقيقية منقّحة + حالات اصطناعية + دلالات نجاح واضحة |
| C | F04/F05/F09: preflight وجاهزية التشغيل وdiagnostics | B لتثبيت شكل النتائج | سجل لكل run، أسباب فشل قابلة للتصرف، واختبارات بدون model calls حيث يمكن |
| D | اختيار مسار Gemini وحفظ default بالمجال الصحيح | F06/F07 + B | تصميم يحدد المصادقة ومخزن credentials والموديل؛ smoke test بهوية runtime مثبتة |
| E | مراجعة تحسينات `D` الخاصة بـsupervisor/retention ونشر release جديد | A + اختبارات تغييرات `D` | snapshot جديد، fresh client، hashes فعالة، خطة رجوع |
| F | تحسين زمن التنفيذ وتجربة التوازي والصيانة | بيانات قياس بعد C/E | benchmark قابل للإعادة وتحسينات محددة |

- [ ] تحويل كل F إلى issue أو مهمة إصلاح بنفس المعرّف والدليل.
- [ ] نقل probes المناسبة إلى tests فعلية في مستودع الـbridge؛ probes هنا تسجّل العيب ولا تصلحه.
- [ ] عدم اعتبار نشر `D` وحده حلاً لـF01/F02/F03.
- [ ] الحفاظ على اختبار مستقل لكل إصلاح قبل تجميع release.
- [ ] عدم مشاركة الكتابة المتوازية إلى `server.js` نفسه؛ يمكن بحث/تصميم المهام بالتوازي ثم دمج edits تسلسلياً أو في worktrees منفصلة.

لا نوصي بإزالة Git safety checks أو السماح بكل plugins أو تحويل `plan/general` إلى permissive agents لمعالجة سرعة الاستخدام.

## 8. سجل التحقق والأدلة القابلة للإعادة

| الفحص | النتيجة | حدود الإثبات |
| --- | --- | --- |
| `get_opencode_bridge_status` | healthy، OpenCode `1.17.13`، Terra high، plugins معطلة | فحص إعداد/اكتشاف؛ لا يثبت استجابة المزوّد |
| preflight ثم run في non-Git | قبول ثم `git_state_required` | reproducer حي لـF04؛ لا model call |
| preflight لـ`plan` و`general` | رفض بصلاحيات غير آمنة / قابلية كتابة | يثبت فعالية الحماية وأثر اختيار agent |
| مراجعة مستقلة عبر MCP | `agent_empty_final_response`؛ exit 0؛ لا تغييرات مكتشفة | لم تنتج مراجعة مستقلة مكتملة؛ لم نعتمد على رأي agent غير متاح |
| diagnose بعد المراجعة | jobs وfailedJobs يساويان 0 | يثبت فرق مسار direct عن سجل queue |
| probes على `R` و`D` | اكتملت بخروج 0، و5 positive controls لكل مصدر؛ أعادت إنتاج الملاحظات | خروج probe 0 يعني نجاح تنفيذ التشخيص، **لا** أن العيوب محلولة |
| `npm test` في `R` | passed، exit 0؛ syntax checks وrelease builder وfresh-healthcheck وTUI smoke و`Self tests passed.` | اختبارات المصدر الموجودة، مع state مؤقت منفصل؛ لا تثبت تغطية كل الحالات الجديدة |
| `npm run test:concurrency` في `R` | passed، exit 0: `Cross-process MCP concurrency stress passed.` | fixture يستخدم Fake OpenCode؛ لا يثبت توازي طلبات Gemini الفعلية |

الملفات المرافقة:

- [probes.mjs](probes.mjs): يشغّل دوالاً مستخرجة من المصدر داخل VM محدودة، بمدخلات اصطناعية، دون استيراد الخادم أو تعديل ملفاته.
- [probes-release.json](evidence/probes-release.json): النتائج وأرقام الدوال وhash النسخة المشغلة.
- [probes-development.json](evidence/probes-development.json): النتائج نفسها على نسخة التطوير، بما يشمل التعديلات غير committed.
- [live-observations.json](evidence/live-observations.json): ملاحظات MCP المنقّحة وتقرير التجربة الفاشلة.

لإعادة probes من مجلد مشروع الشطرنج:

```powershell
node docs/mcp-orchestrator-audit/probes.mjs 'C:\Users\10User\codex-opencode-mcp-releases\server-3e828b6c-20260813-lock-origin\server.js'
node docs/mcp-orchestrator-audit/probes.mjs 'C:\Users\10User\codex-opencode-mcp\server.js'
```

تعيد probes قراءة الدوال نفسها من `server.js`؛ ليست نسخة من implementation. استخراج الدوال يعتمد على حدود declarations في النسختين المفحوصتين، وقد يحتاج تحديثاً إذا تغير شكل المصدر. لا يُستخدم هذا السكربت لتحميل مصدر غير موثوق.

لإعادة suite المعزولة من مجلد `R`، وفق scripts الفعلية في `package.json`:

```powershell
$auditStateRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('mcp-orchestrator-audit-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $auditStateRoot | Out-Null
$env:CODEX_OPENCODE_STATE_DIR = $auditStateRoot
$env:CODEX_OPENCODE_QUEUE_MODE = 'memory'
npm test
npm run test:concurrency
```

المتغيرات أعلاه للعملية الحالية. شُغّلت أوامر الدراسة في عمليات shell منفصلة. لم يتغير config الدائم. اختبارات suite قد تنشئ fixtures مؤقتة وتُنظفها؛ لا تُوجّه state الاختبار إلى قاعدة التشغيل الفعلية.

ما لم يُتحقق منه: benchmark ممثل، نجاح Gemini عبر MCP، contractor حي، تحديث/refresh OAuth، crash حقيقي للخادم المنشور مع أعمال المستخدم، قراءة أسرار حقيقية، اعتماد كل تعديلات `D`، أو نشر release جديد. لم تُشغّل `test:e2e` و`test:e2e:contractor` التي تتضمن نموذجاً حياً. نتيجة محاولة `npm test` الأولى لم تُسترجع بعد انقطاع الجلسة، لذلك أُعيد الفحص ولم تُحتسب المحاولة الأولى نجاحاً.

## 9. خريطة المصادر

الروابط المحلية التالية تشير إلى النسخ المحددة في القسم 2، وقد تحتاج تعديل root عند نقل الدراسة لجهاز آخر. ثبّت المصدر بالـhash قبل الاعتماد على أرقام الأسطر.

| المصدر | ما يدعمه |
| --- | --- |
| [إعداد Codex MCP][config] و[إعداد release][release-config] | ما يُشغَّل فعلياً ومصدر model defaults |
| [ملف orchestrator][orchestrator-profile] | صلاحيات التخطيط وقائمة أوامر shell |
| [بيئة worker][env-r] | تعطيل project config وDB المؤقت |
| [redaction في R][redaction-r] و[في D][redaction-d] | F01 |
| [parser][events-r] و[classifier][classifier-r] و[run evidence][run-evidence-r] | F02/F03/F05 |
| [preflight][preflight-r] و[Git root][git-root-r] و[HEAD][head-r] | F04 |
| [واجهة run][run-tool-r] و[CLI args][cli-args-r] | F06/F09 |
| [plugin verifier][plugins-r] و[immutable mode][immutable-r] | F07 |
| [snapshot validation][snapshots-r] | F10 وحدود attribution |
| [retention القديم][retention-r] و[الجديد][retention-d] | F11 |
| [spawn القديم][spawn-r] و[supervisor الجديد][supervisor-d] | F12 |
| [package.json في D][package-d] | أوامر وفحوص أضيفت في التطوير |
| [OpenCode CLI][opencode-cli]، [Config][opencode-config]، [Permissions][opencode-permissions] | المعنى الرسمي للـflags والإعداد والصلاحيات؛ تمت مراجعة وثائق v1، لا خلطها بصياغة v2 |
| [GitHub authentication][github-token-doc] | البوادئ الصحيحة لأنواع tokens |

[config]: <C:/Users/10User/.codex/config.toml:283>
[release-config]: <C:/Users/10User/codex-opencode-mcp-releases/server-3e828b6c-20260813-lock-origin/opencode/opencode.jsonc:1>
[orchestrator-profile]: <C:/Users/10User/codex-opencode-mcp-releases/server-3e828b6c-20260813-lock-origin/opencode/agents/mcp-orchestrator.md:1>
[env-r]: <C:/Users/10User/codex-opencode-mcp-releases/server-3e828b6c-20260813-lock-origin/server.js:418>
[redaction-r]: <C:/Users/10User/codex-opencode-mcp-releases/server-3e828b6c-20260813-lock-origin/server.js:633>
[redaction-d]: <C:/Users/10User/codex-opencode-mcp/server.js:784>
[events-r]: <C:/Users/10User/codex-opencode-mcp-releases/server-3e828b6c-20260813-lock-origin/server.js:2872>
[classifier-r]: <C:/Users/10User/codex-opencode-mcp-releases/server-3e828b6c-20260813-lock-origin/server.js:3853>
[run-evidence-r]: <C:/Users/10User/codex-opencode-mcp-releases/server-3e828b6c-20260813-lock-origin/server.js:4124>
[preflight-r]: <C:/Users/10User/codex-opencode-mcp-releases/server-3e828b6c-20260813-lock-origin/server.js:8208>
[git-root-r]: <C:/Users/10User/codex-opencode-mcp-releases/server-3e828b6c-20260813-lock-origin/server.js:4469>
[head-r]: <C:/Users/10User/codex-opencode-mcp-releases/server-3e828b6c-20260813-lock-origin/server.js:6063>
[execute-r]: <C:/Users/10User/codex-opencode-mcp-releases/server-3e828b6c-20260813-lock-origin/server.js:10569>
[run-tool-r]: <C:/Users/10User/codex-opencode-mcp-releases/server-3e828b6c-20260813-lock-origin/server.js:8456>
[cli-args-r]: <C:/Users/10User/codex-opencode-mcp-releases/server-3e828b6c-20260813-lock-origin/server.js:3803>
[plugins-r]: <C:/Users/10User/codex-opencode-mcp-releases/server-3e828b6c-20260813-lock-origin/server.js:2388>
[immutable-r]: <C:/Users/10User/codex-opencode-mcp-releases/server-3e828b6c-20260813-lock-origin/server.js:18213>
[snapshots-r]: <C:/Users/10User/codex-opencode-mcp-releases/server-3e828b6c-20260813-lock-origin/server.js:10968>
[retention-r]: <C:/Users/10User/codex-opencode-mcp-releases/server-3e828b6c-20260813-lock-origin/server.js:158>
[retention-d]: <C:/Users/10User/codex-opencode-mcp/server.js:163>
[spawn-r]: <C:/Users/10User/codex-opencode-mcp-releases/server-3e828b6c-20260813-lock-origin/server.js:755>
[supervisor-d]: <C:/Users/10User/codex-opencode-mcp/bin/process-supervisor.js:1>
[package-d]: <C:/Users/10User/codex-opencode-mcp/package.json:1>
[opencode-cli]: https://opencode.ai/docs/cli/
[opencode-config]: https://opencode.ai/docs/config/
[opencode-permissions]: https://opencode.ai/docs/permissions/
[github-token-doc]: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github
