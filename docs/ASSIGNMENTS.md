# Graded assignments

MR Studio can give every student their own version of a problem set and grade the answers on your side.
This page is for the instructor. It covers making an assignment, what students do, grading a folder of
submissions, and (optionally) grading in the private course repository with GitHub Actions.

## Be clear about what this does and does not protect

- **The answers are not secret.** MR Studio is open source, so the problem generators are public. A
  student who knows Python can recompute the answer to any problem, and the Practice tab's *Problems* page shows
  worked solutions for any seed. Do not use this for exams where that matters.
- **What it does give you:** every student gets different numbers (the seed comes from the assignment code
  and their student ID), so copying a classmate's numbers earns nothing. The Assignment panel itself only shows
  the problem statements, never the solutions, and it does not tell the student whether an answer is right.
- **Grading happens on your side.** The submission file holds only what the student typed. You (or a CI job
  you control) regenerate each student's problems from the code and compare.
- **The fingerprint in a submission is not a signature.** It is a checksum of the code, the student ID and the
  answers. It catches a file that was edited by hand after saving or saved for the wrong assignment. Anyone who
  has the code can recompute it, so it does not prove who made the file.
- **The "late" flag uses the student's computer clock.** Use the upload time in your LMS for deadlines.
- **Grade with the same MR Studio version the students used.** Problems are fixed by (topic, seed,
  difficulty) and the seeds are never changed between versions, but a future fix to a generator could change
  a number. Each submission records `app_version`; grade with that release (see the CI sketch below).

## 1. Make an assignment (instructor)

From the `MR-Studio` folder, with the project's Python (`.venv/bin/python`):

```bash
# which topics exist and which quantities students type for each difficulty
.venv/bin/python tools/assignment.py topics

# three problems, due at the end of 20 Nov 2026
.venv/bin/python tools/assignment.py new --id hw3 --title "Homework 3" --due 2026-11-20 \
    --out hw3.code fk:2 jacobian:2:Jb,singular control:1
```

Each item is `topic[:difficulty[:quantity,quantity…]]`:

- `difficulty` is 1 (basic), 2 (standard, the default) or 3 (challenge).
- The optional quantity list limits what students must type (default: every numeric answer, e.g. a 4×4
  matrix `T` for forward kinematics). `jacobian:2:Jb,singular` asks only for the body Jacobian and the
  singular yes/no.
- "mixed" is not allowed: every student must get the same kind of problem.
- A few quantities depend on the student's numbers: the control problems ask for the damped frequency
  `wd` when the response is underdamped and for the two real `roots` when it is overdamped. Each student is
  asked for what their own problem has. Roots and eigenvalues may be typed in any order.

The command prints the **assignment code** (one long line of letters, digits, `-` and `_`) and with `--out`
also saves it to a file. **Keep that code or file**: it contains a random salt, and grading needs exactly the
same code. Making a second code with the same id gives every student different problems.

Useful checks:

```bash
.venv/bin/python tools/assignment.py show hw3.code          # id, title, due date, items (no answers)
.venv/bin/python tools/assignment.py key hw3.code 20231234  # one student's seeds and answers (keep private)
```

Tolerance: an answer counts as correct when the largest difference over its entries is at most
`1 % of the largest reference entry + 0.0001` (options `--rtol` and `--atol` change this for the whole
assignment). In practice students should give 3–4 significant figures, or 4 decimals for numbers near zero.
Angles are in radians and everything is in SI units, as in the worked solutions. Each problem is worth one
point, split evenly over its quantities.

## 2. What students do

1. Open the **Practice** tab, then the **Assignment** page.
2. Paste the code, type their student ID (letters, digits, `.`, `_`, `-`; no spaces) and press
   **Load assignment**.
3. Pick a problem in the list. Its statement appears in the middle of the window. Type the final numbers in
   the answer fields: vectors and matrices as numbers separated by commas or spaces, matrices row by row. A red
   border only means the format or the count of numbers is off; it says nothing about correctness.
4. Press **Save submission**. The file is written to
   `Documents/MR Studio/submissions/<student id>_<assignment id>.json` (the panel shows the full path and
   **Show folder** opens it).
5. Upload that file to the LMS assignment (Moodle, Blackboard, Google Classroom, …).

Typed answers are kept between sessions, so a student can close MR Studio and continue later. Saving again
overwrites the file with the latest answers.

Tell students to use the same student ID as on the LMS, because their problems depend on it.

## 3. Grade (instructor)

Download all submissions from the LMS into one folder (subfolders are fine), then:

```bash
.venv/bin/python tools/grade_assignments.py hw3.code downloads/hw3 -o hw3-grades.csv --details hw3-details.json
```

`hw3-grades.csv` has one row per file: `student_id, file, score, max_score, percent, p1 … pN, late,
fingerprint_ok, code_ok, created, app_version, notes`. `hw3-details.json` has the result of every quantity
(correct or not, the error and the tolerance) for feedback or appeals. Files that are not submissions get a
row that says "skipped" instead of stopping the run.

Look at the `notes` column: it flags edited files (fingerprint), files made for another code or assignment,
and a student ID that does not match.

The same functions are available in Python (`mrstudio.assignments.build`, `submit`, `grade`,
`answer_key`) for custom workflows.

## 4. Optional: grading in the private labs repository (GitHub Actions)

The private repository `robotics-track-labs-ci-revised` already runs its lab checks by hand
(`workflow_dispatch`). Grading can run the same way. **Nothing has been copied into that repository**: doing
so is your decision. A suggested layout:

```
robotics-track-labs-ci-revised/
  assignments/
    hw3/
      code.txt            # the assignment code (private: it lets anyone regenerate every student's problems)
      submissions/        # the files downloaded from the LMS
  .github/workflows/grade.yml
```

The grader is not copied: the workflow checks out the public MR Studio repository at the release the students
used and runs its `tools/grade_assignments.py`. Sketch of `.github/workflows/grade.yml`:

```yaml
name: grade an MR Studio assignment
on:
  workflow_dispatch:
    inputs:
      assignment:
        description: "folder under assignments/ (e.g. hw3)"
        required: true
      mrstudio_ref:
        description: "MR Studio release tag the students used (see app_version in the submissions)"
        required: true
        default: "v1.0.0"
jobs:
  grade:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4                      # this private repo
      - uses: actions/checkout@v4                      # the public app, pinned
        with:
          repository: MIslam1010/mr-studio
          ref: ${{ inputs.mrstudio_ref }}
          path: mr-studio
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install the maths only (no GUI needed for grading)
        run: python -m pip install numpy scipy modern_robotics
      - name: Grade
        run: |
          A=assignments/${{ inputs.assignment }}
          python mr-studio/tools/grade_assignments.py "$A/code.txt" "$A/submissions" \
              -o grades.csv --details details.json
      - uses: actions/upload-artifact@v4               # private to the repo's collaborators
        with:
          name: grades-${{ inputs.assignment }}
          path: |
            grades.csv
            details.json
```

Notes on the sketch:

- Pin the versions: install the same `numpy`, `scipy` and `modern_robotics` versions as in MR Studio's
  `packaging/requirements-lock.txt` for that release, so the numbers are computed exactly as on the students'
  machines.
- Keep it manual (`workflow_dispatch`) and the repository private: `code.txt` and the grades are student data.
- The run passes even when students score low; read the CSV. (`--min-percent N` makes it fail when any
  submission is below N %, if you want a red mark for that.)
- Upload `grades.csv` to the LMS grade book by hand, or import it.
