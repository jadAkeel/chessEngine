# External Training Path

## الهدف
هذا المسار مخصص لتدريب النموذج على بيانات خارجية فقط، بدون خلطه مع مسار `self-play` التقليدي.

## ما الذي تغيّر
- `app/training/loop.py` عاد ليكون خاصًا بـ `self-play` فقط.
- تمت إضافة مسار مستقل: `app/training/train_external.py`.
- تمت إضافة واجهة تشغيل: `app/cli/train_external.py`.
- تمت إضافة قسم `external` في الإعدادات، وملف إعداد منفصل: `config/external_training.yaml`.
- نقاط الحفظ الخارجية أصبحت منفصلة باسم prefix واضح مثل:
  - `external_latest_checkpoint.pth`
  - `external_best_model.pth`
- تحميل البيانات الخارجية أصبح يسجل بوضوح:
  - عدد العينات الموجودة في الملف
  - عدد العينات المفحوصة
  - عدد العينات المقبولة
  - عدد العينات المستبعدة بسبب التكرار
  - عدد العينات المستبعدة بسبب القيم/الحالات غير الصالحة
- يتم خلط العينات (`shuffle`) قبل التقسيم.
- يتم اقتطاع جزء للتحقق (`validation split`).
- يوجد benchmark ثابت بعد التدريب لمقارنة النموذج قبل/بعد التدريب الخارجي.

## التشغيل
```bash
python -m app.cli.train_external --config config/external_training.yaml
```

## ملاحظات
- dedup الحالي يعتمد على: `state + move_index + value`.
- filtering الحالي يستبعد الحالات التالية:
  - policy index غير صالح
  - value غير finite
  - state يحتوي NaN/Inf
  - state صفري بالكامل إذا كان `drop_zero_states=true`
- يمكن تعديل سلوك المسار الخارجي كاملًا من قسم `external` بدون التأثير على `self-play`.
