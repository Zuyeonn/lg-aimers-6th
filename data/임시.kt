class EmotionActionTimeFragment : Fragment() {

    private var _binding: FragmentEmotionActionTimeBinding? = null
    private val binding get() = _binding!!

    private lateinit var viewModel: EmotionModel


    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentEmotionActionTimeBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        // 시스템 바(상태바, 네비게이션바) 공간 자동 조정
        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        // viewModel 가져오기
        viewModel = ViewModelProvider(requireActivity()).get(EmotionModel::class.java)

        setupHourPicker()
        setupMinutePicker()

        // 상태바 업데이트 (감정 비율 기반)
        viewModel.emotionRatios.observe(viewLifecycleOwner) { updateStatusBar(it) }

        // 캐릭터 업데이트 (최근 감정 기반)
        viewModel.recentEmotion.observe(viewLifecycleOwner) { updateCharacter(it) }

        // 감정에 따른 말풍선 색상 적용
        viewModel.colorResId.observe(viewLifecycleOwner) { colorResId ->
            binding.timeFl.backgroundTintList =
                ContextCompat.getColorStateList(requireContext(), colorResId)
        }

        // "확인" 버튼 클릭 시 -> 데이터 저장 + Fragment 이동
        binding.timeConfirmTv.setOnClickListener {
            val selectedHour = binding.timeHourNp.value
            val selectedMinute = binding.timeMinuteNp.displayedValues[binding.timeMinuteNp.value].toInt()
            // ViewModel에 저장
            viewModel.setActivityTime(selectedHour, selectedMinute)
            // 데이터 전달하면서 Fragment 이동
            navigateToTime2(selectedHour, selectedMinute)
        }
    }

    // 상태바 업데이트 (감정 비율에 따른 색상 적용)
    private fun updateStatusBar(emotionRatios: Map<String, Float>) {
        val sortedRatios = viewModel.getSortedEmotionRatios()
        val sortedColors = sortedRatios.mapNotNull { (emotion, _) ->
            viewModel.getEmotionColor(emotion)?.let { ContextCompat.getColor(requireContext(), it) }
        }
        if (sortedColors.isNotEmpty()) {
            val dominantColor = sortedColors.first()
            binding.statusBar.setColorFilter(dominantColor, PorterDuff.Mode.SRC_IN)
        }
    }

    // 최근 감정 기반 캐릭터 변경
    private fun updateCharacter(emotion: String) {
        val characterResId = viewModel.getCharacterForEmotion(emotion)
        binding.iconIv.setImageResource(characterResId)
    }


    private fun navigateToTime2(selectedHour: Int, selectedMinute: Int) {
        val fragment = EmotionActionTimeFragment2().apply {
            arguments = Bundle().apply {
                putInt("HOUR", selectedHour)
                putInt("MINUTE", selectedMinute)
            }
        }
        requireActivity().supportFragmentManager.beginTransaction()
            .replace(R.id.main_container, fragment)
            .addToBackStack(null)
            .commit()
    }


    private fun setupHourPicker() {
        binding.timeHourNp.apply {
            minValue = 0
            maxValue = 24
            wrapSelectorWheel = false
        }
    }
    private fun setupMinutePicker() {
        val minuteValues = arrayOf("00", "05", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60")
        binding.timeMinuteNp.apply {
            minValue = 0
            maxValue = minuteValues.size - 1
            displayedValues = minuteValues
            wrapSelectorWheel = true
        }
    }


    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
