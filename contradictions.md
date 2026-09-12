# ClauseCheck Project: Documented Contradictions Analysis
Author: ClauseCheck Technical Writing Team
Date: September 2026

This document details the three intentional contradictions planted within the ClauseCheck corpus. These conflicts simulate real-world policy desynchronization that occurs when different university departments (e.g., Academic Senate, Finance, Student Affairs) independently draft and update administrative regulations.

## Contradiction 1: Attendance Policy Thresholds

**Files Involved:** `corpus/attendance_policy.md`

**Nature of Conflict:** Three different clauses provide mutually exclusive attendance requirements for examination eligibility.

**Specific Clauses:**
- **Clause 3.1 (Standard Requirement):** "Students must maintain at least 75% attendance to be eligible to sit the end-semester examination for any given course."
- **Clause 5.2 (Medical Exemption):** "Students with documented medical absences shall be eligible for examination provided they maintain a minimum of 60% attendance, with the shortfall covered by approved medical certificates."
- **Clause 7.1 (Committee Powers):** "The Academic Standing Committee... is empowered to waive attendance requirements entirely in exceptional circumstances, such as severe life-threatening illnesses, national emergencies, or other extraordinary events beyond the student's control, overriding any other minimum threshold."

**Analysis:** This is a genuine contradiction of thresholds. Clause 3.1 states a hard 75% rule. Clause 5.2 lowers this to 60% for medical reasons. However, Clause 7.1 grants a committee the power to waive the requirement *entirely* for severe illnesses. If a student has a severe illness that keeps them at 30% attendance, Clause 5.2 implies they are strictly ineligible (because they are below the 60% minimum for medical cases), but Clause 7.1 states the committee can waive it entirely for that exact reason (severe illness). This creates a conflict regarding what happens when a student falls below 60% due to severe medical issues.

## Contradiction 2: Late Payment Penalties for Overdue Tuition Dues

**Files Involved:** `corpus/fee_deadlines_table.md`

**Nature of Conflict:** Two different sections provide conflicting percentage rates for late-payment penalties applied to overdue tuition balances.

**Specific Clauses:**
- **Section 4, Clause 4.2 (Late Payment Penalties):** "Late fee is calculated at 5% of outstanding dues per week of delay. This penalty is strictly enforced and is automatically calculated and applied by the billing system at 11:59 PM every Friday following the payment deadline."
- **Section 8, Clause 8.1 (Delinquent Accounts and Collections):** "Any tuition balance remaining unpaid following the published semester deadline will accrue a mandatory late-payment surcharge assessed at 2% of the total outstanding balance per week of delay until settled in full."

**Analysis:** This is a direct numerical contradiction for the exact same event. A student who has an overdue tuition balance past the semester deadline faces two conflicting penalty schedules within the same regulation document: Section 4.2 mandates a 5% weekly late fee, whereas Section 8.1 specifies a 2% weekly late fee for unpaid balances. Neither clause references or overrides the other, creating an irreconcilable conflict over what percentage surcharge is legally assessed.

## Contradiction 3: Hostel Guest Policy - Overnight Stays

**Files Involved:** `corpus/hostel_handbook.md`

**Nature of Conflict:** Two different sections within the same handbook give mutually exclusive directives regarding whether non-residents can stay overnight in the hostel facilities.

**Specific Clauses:**
- **Section 3, Clause 3.4 (Guest & Visitor Policy):** "All visitors must vacate the hostel premises by 9:00 PM. No overnight stays by non-residents are permitted under any circumstances."
- **Section 8, Clause 8.4 (Common Room & Recreation Facilities):** "Residents may host overnight guests in common-area guest rooms with prior written approval from the Warden, provided the guest registers at the hostel office by 6:00 PM and vacates by 11:00 AM the following day."

**Analysis:** This is a direct, undeniable contradiction. Section 3 uses absolute language ("under any circumstances") to explicitly ban overnight stays by non-residents. Conversely, Section 8 provides a structured procedure (approval from Warden, 6:00 PM registration, 11:00 AM checkout) explicitly permitting overnight stays for guests in common-area guest rooms. A resident reading Section 3 would believe it is impossible to host a friend or family member overnight. A resident reading Section 8 would understand there is an established process to do exactly that. This simulates a common error where the general rules section is not updated to reflect new special amenities (like guest rooms) added to a facility.

*End of Analysis Document*
